# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging
from abc import abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Reshape
from tensorflow.keras.losses import Huber

from spivak.data.dataset import Dataset, Task, LabelsFromTaskDict, INDEX_LABELS
from spivak.models.assembly.head import TrainerHeadInterface, \
    get_chunk_targets_and_weights, get_tf_targets_and_weights_mapper, \
    PredictorHeadInterface, compute_weights_from_counts, create_head_stack
from spivak.models.assembly.layers import convolution_wrapper, \
    NODE_PENULTIMATE, NODE_SUFFIX_CONVOLUTION, Nodes, INITIALIZER_FOR_SIGMOID
from spivak.models.assembly.weight_creator import WeightCreatorInterface, \
    create_frame_range

VERY_LARGE_DELTA = 10.0
NEGATIVE_DELTA = VERY_LARGE_DELTA
DELTA_OUTPUT_DIMENSION = 1
# The 2 below corresponds to 1 for the actual delta targets, plus 1 for the
# weight inputs.
DELTA_TARGET_DIMENSION = 2


class DeltaPredictorHeadInterface(PredictorHeadInterface):

    @property
    @abstractmethod
    def radius(self) -> float:
        pass

    @property
    @abstractmethod
    def num_chunk_frames(self) -> int:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass


class DeltaTrainerHead(TrainerHeadInterface):

    # For each time instant and class, delta corresponds to the parameterized
    # time displacement from that instant to the closest label.

    def __init__(
            self, delta_predictor_head: DeltaPredictorHeadInterface,
            weight_creator: WeightCreatorInterface) -> None:
        self._radius = delta_predictor_head.radius
        self._num_chunk_frames = delta_predictor_head.num_chunk_frames
        self._num_classes = delta_predictor_head.num_classes
        self._delta_predictor_head = delta_predictor_head
        self._weight_creator = weight_creator

    def video_targets(
            self, video_labels_from_task: LabelsFromTaskDict) -> np.ndarray:
        video_labels = video_labels_from_task[Task.SPOTTING][INDEX_LABELS]
        non_zeros = np.nonzero(video_labels)
        shape = video_labels.shape
        delta_targets = create_delta_targets(non_zeros, self._radius, shape)
        weight_inputs = self._weight_creator.video_weight_inputs(
            video_labels, delta_targets)
        return np.stack([delta_targets, weight_inputs], axis=2)

    def tf_chunk_targets_mapper(
            self, video_targets_and_weight_inputs, task: Task):
        valid_task = DeltaTrainerHead._valid_task(task)
        return get_tf_targets_and_weights_mapper(
            video_targets_and_weight_inputs, self._num_chunk_frames,
            self._weight_creator, valid_task)

    def chunk_targets(
            self, video_targets_and_weight_inputs, start: int,
            mask: np.ndarray, task: Task) -> np.ndarray:
        valid_task = DeltaTrainerHead._valid_task(task)
        return get_chunk_targets_and_weights(
            video_targets_and_weight_inputs, start, mask,
            self._num_chunk_frames, self._weight_creator, valid_task)

    @property
    def video_targets_shape(self):
        return None, self._num_classes, DELTA_TARGET_DIMENSION

    @property
    def predictor_head(self):
        return self._delta_predictor_head

    @property
    def supports_mixup(self) -> bool:
        return False

    @staticmethod
    def _valid_task(task: Task) -> bool:
        return task == Task.SPOTTING


class DeltaPredictorHead(DeltaPredictorHeadInterface):

    def __init__(
            self, name: str, radius: float, delta_loss: "DeltaLoss",
            num_chunk_frames: int, num_classes: int, weight_decay: float,
            batch_norm: bool, dropout_rate: float, width: int,
            num_head_layers: int, zero_init: bool) -> None:
        self._name = name
        self._delta_loss = delta_loss
        self._radius = radius
        self._num_chunk_frames = num_chunk_frames
        self._num_classes = num_classes
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._dropout_rate = dropout_rate
        self._width = width
        self._num_head_layers = num_head_layers
        self._zero_init = zero_init

    def create_tensor(self, nodes: Nodes) -> Tensor:
        delta_tensor_input, delta_tensor_window_size = create_head_stack(
            self._name, self._weight_decay, self._batch_norm,
            self._dropout_rate, self._width, self._num_head_layers,
            nodes[NODE_PENULTIMATE])
        return _create_delta_tensor(
            self._name, self._num_chunk_frames, self._num_classes, 1,
            self._weight_decay, self._batch_norm, self._dropout_rate,
            delta_tensor_window_size, self._zero_init, delta_tensor_input)

    def post_process(self, delta):
        result = _post_process_delta(delta, self._radius)
        return result

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dimension(self):
        return DELTA_OUTPUT_DIMENSION

    @property
    def loss(self):
        return self._delta_loss.loss

    @property
    def loss_name(self):
        return self._delta_loss.name

    @property
    def loss_weight(self):
        return self._delta_loss.weight

    @property
    def num_chunk_frames(self):
        return self._num_chunk_frames

    @property
    def radius(self):
        return self._radius


class DeltaLoss:

    def __init__(self, weight, base_loss):
        # Note: don't change this function's name, since the name is
        # currently used when loading the models. Should eventually fix this
        # issue.
        def delta_loss(targets_and_weights, predictions):
            targets = targets_and_weights[:, :, :, 0:1]
            weights = targets_and_weights[:, :, :, 1]
            return _create_delta_loss(targets, predictions, weights, base_loss)

        self.weight = weight
        self.loss = delta_loss
        self.name = "delta_loss"


class DeltaWeightCreator(WeightCreatorInterface):

    def __init__(
            self, dataset: Dataset, delta_radius: float,
            positive_weight: float, class_weights: np.ndarray) -> None:
        num_classes = dataset.num_classes_from_task[Task.SPOTTING]
        positive_counts_per_class = np.zeros(num_classes)
        negative_counts_per_class = np.zeros(num_classes)
        for video_datum in dataset.video_data:
            video_labels = video_datum.labels(Task.SPOTTING)
            video_positive_counts, video_negative_counts = \
                DeltaWeightCreator._video_counts(video_labels, delta_radius)
            positive_counts_per_class += video_positive_counts
            negative_counts_per_class += video_negative_counts
        positive_weight_per_class, negative_weight_per_class = \
            compute_weights_from_counts(
                positive_counts_per_class, negative_counts_per_class,
                positive_weight)
        # Element-wise multiplication to take into account the class_weights.
        self._positive_weight_per_class = (
                class_weights * positive_weight_per_class)
        self._negative_weight_per_class = (
                class_weights * negative_weight_per_class)
        logging.info("Delta positive_weight_per_class")
        logging.info(self._positive_weight_per_class)
        logging.info("Delta negative_weight_per_class")
        logging.info(self._negative_weight_per_class)

    def video_weight_inputs(self, video_labels, video_targets):
        return np.where(
            video_targets != NEGATIVE_DELTA, self._positive_weight_per_class,
            self._negative_weight_per_class)

    def tf_chunk_weights(self, chunk_weight_inputs):
        return chunk_weight_inputs

    def chunk_weights(self, chunk_weight_inputs):
        return chunk_weight_inputs

    @staticmethod
    def _video_counts(
            video_labels: np.ndarray,
            delta_radius: float) -> Tuple[np.ndarray, np.ndarray]:
        non_zeros = np.nonzero(video_labels)
        shape = video_labels.shape
        video_delta_targets = create_delta_targets(
            non_zeros, delta_radius, shape)
        video_positive_counts_per_class = np.sum(
            video_delta_targets != NEGATIVE_DELTA, axis=0)
        video_negative_counts_per_class = (
                len(video_delta_targets) - video_positive_counts_per_class)
        return video_positive_counts_per_class, video_negative_counts_per_class


def create_delta_targets(non_zeros, radius: float, shape):
    frame_indexes, class_indexes = non_zeros
    num_frames, num_classes = shape
    deltas = np.full((num_frames, num_classes), VERY_LARGE_DELTA, dtype='float')
    for frame_index, class_index in zip(frame_indexes, class_indexes):
        # Set the reparameterized delta values.
        delta_range = create_frame_range(frame_index, radius, num_frames)
        target_reparameterized_deltas = (
                (frame_index - delta_range) / radius)
        # Only set the new values if they are smaller than what is already
        # there, so the closest positive example wins.
        delta_selection = np.abs(target_reparameterized_deltas) < np.abs(
            deltas[delta_range, class_index])
        delta_indexes = delta_range[delta_selection]
        deltas[delta_indexes, class_index] = \
            target_reparameterized_deltas[delta_selection]
    return deltas


def create_huber_base_loss(huber_delta: float):
    def huber_base_loss(y_true, y_pred):
        # The y_true and y_pred are expected to be between -1.0 and 1.0. They
        # will get mapped to time intervals according to different radius. I
        # decided to not normalize for the radius when setting huber_delta,
        # since I imagine larger radius will naturally lead to larger errors,
        # and most outliers will be false positive/negatives that are
        # completely wrong, in which case normalizing for the radius doesn't
        # seem great.
        huber = Huber(
            delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)
        # The division is such that the derivative of the loss for
        # differences larger than huber_delta will be 1.0, independently of
        # huber_delta itself.
        return huber(y_true, y_pred) / huber_delta
    return huber_base_loss


def _create_delta_tensor(
        name, num_chunk_frames, num_classes, num_radii, weight_decay,
        batch_norm, dropout_rate, window_size, zero_init: bool,
        penultimate: Tensor) -> Tensor:
    # Define a parameterized version of the deltas. In the application code,
    # it will be multiplied by some window size in order to translate it into
    # an actual time or frame delta. Note we use a tanh activation on it,
    # since we want to allow the delta floats to be in [-1, 1].
    convolution_name = f"{name}_{NODE_SUFFIX_CONVOLUTION}"
    if zero_init:
        kernel_initializer = tf.keras.initializers.Zeros()
    else:
        kernel_initializer = INITIALIZER_FOR_SIGMOID
    deltas_convolution = convolution_wrapper(
        penultimate, num_classes * num_radii, (window_size, 1), (1, 1),
        "same", weight_decay, batch_norm, dropout_rate,
        kernel_initializer, name=convolution_name, activation="tanh")
    reshape = Reshape((num_chunk_frames, num_classes, num_radii), name=name)
    return reshape(deltas_convolution)


def _create_delta_loss(targets, predictions, weights, base_loss) -> Tensor:
    clipped_targets = tf.clip_by_value(targets, -1.0, 1.0)
    all_losses = base_loss(clipped_targets, predictions)
    return K.mean(weights * all_losses, axis=(1, 2))


def _post_process_delta(delta, radius: float):
    return np.squeeze(radius * delta, axis=0)
