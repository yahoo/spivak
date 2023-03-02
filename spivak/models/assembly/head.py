# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from abc import ABCMeta, abstractmethod
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from spivak.data.dataset import LabelsFromTaskDict, Task
from spivak.models.assembly.layers import Nodes, \
    NODE_SUFFIX_HEAD_STACK_CONVOLUTION, convolution_wrapper, \
    INITIALIZER_FOR_RELU
from spivak.models.assembly.weight_creator import WeightCreatorInterface

HEAD_STACK_WIDTH_FACTOR = 16


class PredictorHeadInterface(metaclass=ABCMeta):

    @abstractmethod
    def create_tensor(self, nodes: Nodes) -> Tensor:
        """This method will not be called when a models is loaded from disk.
        It is only used when creating a new models that has not yet been
        saved."""
        pass

    @abstractmethod
    def post_process(self, network_output):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    @property
    @abstractmethod
    def output_dimension(self):
        pass

    # We currently need loss and loss_name in order to load the models from
    # disk, but maybe that could be changed later.
    @property
    @abstractmethod
    def loss(self):
        pass

    @property
    @abstractmethod
    def loss_name(self):
        pass

    @property
    @abstractmethod
    def loss_weight(self):
        pass


class TrainerHeadInterface(metaclass=ABCMeta):

    @abstractmethod
    def video_targets(
            self, video_labels_from_task: LabelsFromTaskDict) -> np.ndarray:
        pass

    @abstractmethod
    def tf_chunk_targets_mapper(self, video_targets_and_weights, task: Task):
        pass

    @abstractmethod
    def chunk_targets(
            self, video_targets_and_weights, start: int, mask: np.ndarray,
            task: Task) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def video_targets_shape(self):
        pass

    @property
    @abstractmethod
    def predictor_head(self):
        pass

    @property
    @abstractmethod
    def supports_mixup(self) -> bool:
        pass


def create_head_stack(
        name: str, weight_decay: float, batch_norm: int, dropout_rate: float,
        width: int, num_head_layers: int, tensor: Tensor) -> Tuple[Tensor, int]:
    if num_head_layers > 1:
        initial_window_sizes = [3] + ([1] * (num_head_layers - 2))
        last_tensor_window_size = 1
    else:
        initial_window_sizes = []
        last_tensor_window_size = 3
    for w, window_size in enumerate(initial_window_sizes):
        convolution_name = f"{name}_{NODE_SUFFIX_HEAD_STACK_CONVOLUTION}_{w}"
        tensor = convolution_wrapper(
            tensor, width * HEAD_STACK_WIDTH_FACTOR, (window_size, 1), (1, 1),
            "same", weight_decay, batch_norm, dropout_rate,
            INITIALIZER_FOR_RELU, name=convolution_name, activation="relu")
    return tensor, last_tensor_window_size


def get_chunk_targets_and_weights(
        video_targets_and_weights, start: int, mask: np.ndarray,
        num_chunk_frames: int, weight_creator: WeightCreatorInterface,
        valid: bool) -> np.ndarray:
    return get_multidimensional_chunk_targets_and_weights(
        video_targets_and_weights, start, mask, num_chunk_frames,
        [weight_creator], valid)


def get_multidimensional_chunk_targets_and_weights(
        video_targets_and_weights, start: int, mask: np.ndarray,
        num_chunk_frames: int, weight_creators: List[WeightCreatorInterface],
        valid: bool) -> np.ndarray:
    chunk_targets_and_weights = slice_chunk_from_video_targets(
        video_targets_and_weights, start, num_chunk_frames)
    # By convention, the weights occupy the second half.
    num_targets_and_weights = chunk_targets_and_weights.shape[2]
    weight_indices = range(
        num_targets_and_weights // 2, num_targets_and_weights)
    if valid:
        for weight_index, weight_creator in zip(
                weight_indices, weight_creators):
            chunk_weight_inputs = chunk_targets_and_weights[:, :, weight_index]
            chunk_weights = weight_creator.chunk_weights(chunk_weight_inputs)
            chunk_weights[mask, :] = 0.0
            chunk_targets_and_weights[:, :, weight_index] = chunk_weights
    else:
        chunk_targets_and_weights[:, :, weight_indices] = 0.0
    return chunk_targets_and_weights


def get_tf_targets_and_weights_mapper(
        video_targets_and_weight_inputs, num_chunk_frames: int,
        weight_creator: WeightCreatorInterface, valid: bool):
    return get_tf_multidimensional_targets_and_weights_mapper(
        video_targets_and_weight_inputs, num_chunk_frames, [weight_creator],
        valid)


def get_tf_multidimensional_targets_and_weights_mapper(
        video_targets_and_weight_inputs, num_chunk_frames: int,
        weight_creators: List[WeightCreatorInterface], valid: bool):

    @tf.function
    def create_chunk_targets_and_weights(chunk_start_and_mask):
        chunk_start, chunk_mask = chunk_start_and_mask
        # Note that, depending on the minimum valid chunk size allowed,
        # chunk_targets_and_weight_inputs might have less than
        # num_chunk_frames frames in it in the case that chunk_start is
        # close to the end of the video. chunk_mask will have a matching
        # number of frames. Later below, we pad the result in order to be
        # able to batch targets and weights from different chunks.
        chunk_targets_and_weight_inputs = video_targets_and_weight_inputs[
                                    chunk_start:chunk_start + num_chunk_frames]
        num_classes = chunk_targets_and_weight_inputs.shape[1]
        num_targets_and_weights = chunk_targets_and_weight_inputs.shape[2]
        num_targets = num_weights = num_targets_and_weights // 2
        chunk_targets = chunk_targets_and_weight_inputs[:, :, 0:num_targets]
        chunk_weight_inputs = chunk_targets_and_weight_inputs[
                              :, :, num_targets:num_targets_and_weights]
        chunk_weights_array = tf.TensorArray(tf.float32, size=num_weights)
        for w, weight_creator in enumerate(weight_creators):
            chunk_weights = weight_creator.tf_chunk_weights(
                chunk_weight_inputs[:, :, w])
            chunk_weights_array = chunk_weights_array.write(w, chunk_weights)
        chunk_weights_unmasked = tf.transpose(
            chunk_weights_array.stack(), perm=[1, 2, 0])
        mask_weights = tf.cast(tf.logical_not(chunk_mask), tf.float32)
        expanded_mask_weights = tf.expand_dims(
            tf.expand_dims(mask_weights, -1), -1)
        tiled_mask_weights = tf.tile(
            expanded_mask_weights, (1, num_classes, num_weights))
        chunk_weights = tf.multiply(chunk_weights_unmasked, tiled_mask_weights)
        if not valid:
            chunk_weights = 0.0 * chunk_weights
        chunk_targets_and_weights = tf.concat(
            [chunk_targets, chunk_weights], axis=2)
        return tf_pad_frames_on_right_3d(
            chunk_targets_and_weights, num_chunk_frames)

    return create_chunk_targets_and_weights


def compute_weights_from_counts(
        positive_counts_per_class: np.ndarray,
        negative_counts_per_class: np.ndarray,
        positive_weight: float) -> Tuple[np.ndarray, np.ndarray]:
    total_counts_per_class = (
            positive_counts_per_class + negative_counts_per_class)
    # Inverse positive frequency times desired positive_weight
    positive_weight_per_class = (
            total_counts_per_class * positive_weight /
            positive_counts_per_class
    )
    # Inverse negative frequency times desired negative_weight (1.0 -
    # positive_weight).
    negative_weight_per_class = (
            total_counts_per_class * (1.0 - positive_weight) /
            negative_counts_per_class
    )
    return positive_weight_per_class, negative_weight_per_class


def slice_chunk_from_video_targets(video_targets, start, num_chunk_frames):
    num_video_frames, num_classes, targets_dimensionality = video_targets.shape
    valid_chunk_labels_len = min(num_chunk_frames, num_video_frames - start)
    if valid_chunk_labels_len == num_chunk_frames:
        # In this case, we don't need any padding, so it's probably a bit
        # faster to just directly create the chunk_features.
        chunk_targets = video_targets[start:start + valid_chunk_labels_len]
    else:
        # Here, we first create the chunk_targets with np.zeros, and then
        # fill in the valid part by copying from features.
        chunk_targets = np.zeros(
            (num_chunk_frames, num_classes, targets_dimensionality))
        chunk_targets[0:valid_chunk_labels_len] = \
            video_targets[start:start + valid_chunk_labels_len]
    return chunk_targets


def create_multidimensional_video_weight_inputs(
        weight_creators: List[WeightCreatorInterface], video_labels, targets):
    multiple_weight_inputs = [
        weight_creator.video_weight_inputs(video_labels, targets[:, :, w])
        for w, weight_creator in enumerate(weight_creators)
    ]
    return np.stack(multiple_weight_inputs, axis=2)


def tf_pad_frames_on_right_2d(
        sliced_tensor: Tensor, num_chunk_frames) -> Tensor:
    paddings = [
        [0, num_chunk_frames - tf.shape(sliced_tensor)[0]], [0, 0]
    ]
    return tf.pad(sliced_tensor, paddings, mode='CONSTANT', constant_values=0)


def tf_pad_frames_on_right_3d(
        sliced_tensor: Tensor, num_chunk_frames) -> Tensor:
    paddings = [
        [0, num_chunk_frames - tf.shape(sliced_tensor)[0]], [0, 0], [0, 0]
    ]
    return tf.pad(sliced_tensor, paddings, mode='CONSTANT', constant_values=0)
