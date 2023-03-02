# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging

import numpy as np
import tensorflow as tf
from scipy.special import softmax
from tensorflow import Tensor
from tensorflow.keras import backend
from tensorflow.keras.layers import Reshape, Softmax
from tensorflow.keras.losses import CategoricalCrossentropy

from spivak.data.dataset import Dataset, Task, LabelsFromTaskDict, INDEX_LABELS
from spivak.data.soccernet_label_io import \
    segmentation_targets_from_change_labels
from spivak.models.assembly.head import TrainerHeadInterface, \
    get_chunk_targets_and_weights, get_tf_targets_and_weights_mapper, \
    PredictorHeadInterface, create_head_stack
from spivak.models.assembly.layers import convolution_wrapper, \
    NODE_PENULTIMATE, NODE_SUFFIX_CONVOLUTION, Nodes, INITIALIZER_FOR_SOFTMAX, \
    NODE_SUFFIX_LOGITS
from spivak.models.assembly.weight_creator import WeightCreatorInterface

SEGMENTATION_OUTPUT_DIMENSION = 1
SEGMENTATION_TARGET_DIMENSION = 2


class SegmentationTrainerHead(TrainerHeadInterface):

    def __init__(
            self, segmentation_predictor_head: "SegmentationPredictorHead",
            weight_creator: WeightCreatorInterface) -> None:
        self._weight_creator = weight_creator
        self._num_chunk_frames = segmentation_predictor_head.num_chunk_frames
        self._num_classes = segmentation_predictor_head.num_classes
        self._segmentation_predictor_head = segmentation_predictor_head

    def video_targets(
            self, video_labels_from_task: LabelsFromTaskDict) -> np.ndarray:
        video_labels = video_labels_from_task[Task.SEGMENTATION][INDEX_LABELS]
        segmentation_targets = segmentation_targets_from_change_labels(
            video_labels)
        weight_inputs = self._weight_creator.video_weight_inputs(
            video_labels, segmentation_targets)
        return np.stack([segmentation_targets, weight_inputs], axis=2)

    def tf_chunk_targets_mapper(
            self, video_targets_and_weight_inputs, task: Task):
        valid_task = SegmentationTrainerHead._valid_task(task)
        return get_tf_targets_and_weights_mapper(
            video_targets_and_weight_inputs, self._num_chunk_frames,
            self._weight_creator, valid_task)

    def chunk_targets(
            self, video_targets_and_weight_inputs, start: int,
            mask: np.ndarray, task: Task) -> np.ndarray:
        valid_task = SegmentationTrainerHead._valid_task(task)
        return get_chunk_targets_and_weights(
            video_targets_and_weight_inputs, start, mask,
            self._num_chunk_frames, self._weight_creator, valid_task)

    @property
    def video_targets_shape(self):
        return None, self._num_classes, SEGMENTATION_TARGET_DIMENSION

    @property
    def predictor_head(self):
        return self._segmentation_predictor_head

    @property
    def supports_mixup(self) -> bool:
        return True

    @staticmethod
    def _valid_task(task: Task) -> bool:
        return task == Task.SEGMENTATION


class SegmentationPredictorHead(PredictorHeadInterface):

    def __init__(
            self, name: str, num_chunk_frames: int, num_classes: int,
            segmentation_loss: "SegmentationLoss", weight_decay: float,
            batch_norm: bool, dropout_rate: float, width: int,
            num_head_layers: int) -> None:
        self._name = name
        self._num_chunk_frames = num_chunk_frames
        self._num_classes = num_classes
        self._segmentation_loss = segmentation_loss
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._dropout_rate = dropout_rate
        self._width = width
        self._num_head_layers = num_head_layers

    def create_tensor(self, nodes: Nodes) -> Tensor:
        segmentation_tensor_input, last_tensor_window_size = create_head_stack(
            self._name, self._weight_decay, self._batch_norm,
            self._dropout_rate, self._width, self._num_head_layers,
            nodes[NODE_PENULTIMATE])
        return _create_segmentation_tensor(
            self._name, self._num_chunk_frames, self._num_classes,
            self._weight_decay, self._batch_norm, self._dropout_rate,
            last_tensor_window_size, segmentation_tensor_input)

    def post_process(self, segmentations):
        # The output head predicts logits, so we need to apply the softmax
        # here during post-processing. Note the first dimension of
        # segmentation was originally for batch instances, so we get rid of it.
        return softmax(segmentations[0], axis=1)

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dimension(self):
        return SEGMENTATION_OUTPUT_DIMENSION

    @property
    def loss(self):
        return self._segmentation_loss.loss

    @property
    def loss_name(self):
        return self._segmentation_loss.name

    @property
    def loss_weight(self):
        return self._segmentation_loss.weight

    @property
    def num_chunk_frames(self):
        return self._num_chunk_frames


class SegmentationLoss:

    def __init__(self, weight: float, focusing_gamma: float) -> None:

        # Note: don't change this function's name, since the name is
        # currently used when loading the models.
        def segmentation_loss(targets_and_weights, expanded_predictions):
            # keras needs the last axis to be over classes for the cross
            # entropy loss, so we get rid of the last axis here.
            targets = targets_and_weights[:, :, :, 0]
            predictions = expanded_predictions[:, :, :, 0]
            # TODO: eventually, separate weights from targets, as at least
            #  in this case it doesn't make sense for the weights to have the
            #  same shape as the targets.
            # The same weight is repeated for all classes inside
            # targets_and_weights, so we just take class 0 below to get weights
            # to the right shape.
            weights = targets_and_weights[:, :, 0, 1]
            return _create_segmentation_loss(
                targets, predictions, weights, focusing_gamma)

        self.loss = segmentation_loss
        self.name = "segmentation_loss"
        self.weight = weight


class SegmentationWeightCreator(WeightCreatorInterface):

    def __init__(
            self, training_set: Dataset, temperature: float,
            base_class_weights: np.ndarray) -> None:
        class_counts = _class_counts(training_set)
        class_weights_from_counts = _class_weights_from_counts(
            class_counts, temperature)
        self._class_weights = base_class_weights * class_weights_from_counts
        logging.info("SegmentationWeightCreator class_counts")
        logging.info(class_counts)
        logging.info("SegmentationWeightCreator class_weights_from_counts")
        logging.info(class_weights_from_counts)
        logging.info("SegmentationWeightCreator._class_weights")
        logging.info(self._class_weights)

    def video_weight_inputs(self, video_labels, video_targets):
        """Read the targets in video_targets and convert them to the
        appropriate weights."""
        frame_classes = video_targets.argmax(axis=1)
        frame_weights = self._class_weights[frame_classes]
        # We repeat frame_weights with the sole purpose of getting the right
        # resulting shape, so we can later append that to targets.
        num_classes = video_targets.shape[1]
        return np.tile(np.expand_dims(frame_weights, axis=1), (1, num_classes))

    def tf_chunk_weights(self, chunk_weight_inputs):
        return chunk_weight_inputs

    def chunk_weights(self, chunk_weight_inputs):
        return chunk_weight_inputs


def _create_segmentation_loss(
        targets, logit_predictions, weights, focusing_gamma: float) -> Tensor:
    if focusing_gamma < 0.0:
        raise ValueError(
            "Value of gamma should be greater than or equal to zero.")
    cce = CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    cce_losses = cce(targets, logit_predictions)
    # Note that we do not use the alpha here to weigh the different classes.
    # Weighing the classes should optionally be done via the input "weights"
    # parameter.
    weighted_losses = weights * cce_losses
    if focusing_gamma > 0.0:
        # Implement the focusing term of the focal loss, presented by Lin et al.
        # https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
        predictions = Softmax(axis=2)(logit_predictions)
        # p_error below corresponds roughly to 1.0 - p_t in the focal loss
        # paper. Here, it is computed as the total variation distance between
        # the targets and predictions.
        # https://math.stackexchange.com/questions/3415641/total-variation-distance-l1-norm
        p_error = backend.sum(tf.math.abs(targets - predictions), axis=2) / 2.0
        # When p_error is zero, the pow becomes problematic, leading to nan.
        # I'm not sure why.
        p_error_safe = tf.math.maximum(p_error, backend.epsilon())
        modulating_factor = tf.pow(p_error_safe, focusing_gamma)
        weighted_losses = modulating_factor * weighted_losses
    return backend.mean(weighted_losses, axis=1)


def _create_segmentation_tensor(
        name: str, num_frames: int, num_classes: int, weight_decay: float,
        batch_norm, dropout_rate: float, window_size: int,
        penultimate: Tensor) -> Tensor:
    segmentation_name = f"{name}_{NODE_SUFFIX_CONVOLUTION}"
    segmentation_convolution = convolution_wrapper(
        penultimate, num_classes, (window_size, 1), (1, 1), "same",
        weight_decay, batch_norm, dropout_rate, INITIALIZER_FOR_SOFTMAX,
        name=segmentation_name)
    reshaped = Reshape(
        (num_frames, num_classes, 1), name=f"{name}_{NODE_SUFFIX_LOGITS}")(
        segmentation_convolution)
    return reshaped


def _class_weights_from_counts(
        class_counts: np.ndarray, temperature: float) -> np.ndarray:
    # The class weights should be inversely proportional to their frequency.
    inverse_counts = 1.0 / class_counts
    # Apply the temperature to make the class weights be more or less smooth.
    power_inverse_counts = np.power(inverse_counts, 1.0 / temperature)
    # We want the total weight to be preserved. i.e. we want
    # sum_i(class_counts[i] * class_weights[i]) == sum_i(class_counts[i])
    inverse_power_count_total_weight = np.sum(
        class_counts * power_inverse_counts)
    original_total_weight = class_counts.sum()
    class_weights = power_inverse_counts * (
            original_total_weight / inverse_power_count_total_weight)
    return class_weights


def _class_counts(training_set: Dataset) -> np.ndarray:
    class_counts = np.zeros(
        training_set.num_classes_from_task[Task.SEGMENTATION])
    for video_datum in training_set.video_data:
        video_targets = segmentation_targets_from_change_labels(
            video_datum.labels(Task.SEGMENTATION))
        class_counts += video_targets.sum(axis=0)
    return class_counts
