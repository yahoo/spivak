# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging
from typing import Tuple, Optional, List

import numpy as np
import tensorflow as tf
from scipy.special import expit
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Reshape, Activation
from tensorflow.keras.losses import BinaryCrossentropy

from spivak.data.dataset import Dataset, Task, LabelsFromTaskDict, INDEX_LABELS
from spivak.models.assembly.head import TrainerHeadInterface, \
    get_chunk_targets_and_weights, get_tf_targets_and_weights_mapper, \
    PredictorHeadInterface, compute_weights_from_counts, \
    get_tf_multidimensional_targets_and_weights_mapper, \
    get_multidimensional_chunk_targets_and_weights, \
    create_multidimensional_video_weight_inputs, create_head_stack
from spivak.models.assembly.layers import convolution_wrapper, \
    NODE_PENULTIMATE, NODE_SUFFIX_CONVOLUTION, Nodes, INITIALIZER_FOR_SIGMOID, \
    NODE_SUFFIX_LOGITS
from spivak.models.assembly.weight_creator import WeightCreatorInterface, \
    create_frame_range

CONFIDENCE_OUTPUT_DIMENSION = 1
CONFIDENCE_TARGET_DIMENSION = 2
NEGATIVE_CONFIDENCE = 0.0


class ConfidenceTrainerHead(TrainerHeadInterface):

    def __init__(
            self, target_radius: float,
            confidence_predictor_head: "ConfidencePredictorHead",
            weight_creator: WeightCreatorInterface) -> None:
        self._target_radius = target_radius
        self._weight_creator = weight_creator
        self._num_chunk_frames = confidence_predictor_head.num_chunk_frames
        self._num_classes = confidence_predictor_head.num_classes
        self._confidence_predictor_head = confidence_predictor_head

    def video_targets(
            self, video_labels_from_task: LabelsFromTaskDict) -> np.ndarray:
        video_labels = video_labels_from_task[Task.SPOTTING][INDEX_LABELS]
        non_zeros = np.nonzero(video_labels)
        shape = video_labels.shape
        confidence_targets = _create_confidence_targets(
            non_zeros, self._target_radius, shape)
        weight_inputs = self._weight_creator.video_weight_inputs(
            video_labels, confidence_targets)
        return np.stack([confidence_targets, weight_inputs], axis=2)

    def tf_chunk_targets_mapper(
            self, video_targets_and_weight_inputs, task: Task):
        valid_task = ConfidenceTrainerHead._valid_task(task)
        return get_tf_targets_and_weights_mapper(
            video_targets_and_weight_inputs, self._num_chunk_frames,
            self._weight_creator, valid_task)

    def chunk_targets(
            self, video_targets_and_weight_inputs, start: int,
            mask: np.ndarray, task: Task) -> np.ndarray:
        valid_task = ConfidenceTrainerHead._valid_task(task)
        return get_chunk_targets_and_weights(
            video_targets_and_weight_inputs, start, mask,
            self._num_chunk_frames, self._weight_creator, valid_task)

    @property
    def video_targets_shape(self):
        return None, self._num_classes, CONFIDENCE_TARGET_DIMENSION

    @property
    def predictor_head(self):
        return self._confidence_predictor_head

    @property
    def supports_mixup(self) -> bool:
        return True

    @staticmethod
    def _valid_task(task: Task) -> bool:
        return task == Task.SPOTTING


class ConfidencePredictorHead(PredictorHeadInterface):

    def __init__(
            self, name: str, num_chunk_frames: int, num_classes: int,
            confidence_loss: "ConfidenceLoss", weight_decay: float,
            batch_norm: bool, dropout_rate: float, width: int,
            num_head_layers: int,
            confidence_class_counts: Optional["ConfidenceClassCounts"]) -> None:
        self._name = name
        self._num_chunk_frames = num_chunk_frames
        self._num_classes = num_classes
        self._confidence_loss = confidence_loss
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._dropout_rate = dropout_rate
        self._width = width
        self._num_head_layers = num_head_layers
        self._confidence_class_counts = confidence_class_counts

    def create_tensor(self, nodes: Nodes) -> Tensor:
        confidence_tensor_input, last_tensor_window_size = create_head_stack(
            self._name, self._weight_decay, self._batch_norm,
            self._dropout_rate, self._width, self._num_head_layers,
            nodes[NODE_PENULTIMATE])
        return _create_confidence_tensor(
            self._name, self._num_chunk_frames, self._num_classes, 1,
            self._weight_decay, self._batch_norm, self._dropout_rate,
            last_tensor_window_size, confidence_tensor_input,
            self._confidence_class_counts)

    def post_process(self, confidences):
        # First dimension is for batch instances.
        return expit(confidences[0])

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dimension(self):
        return CONFIDENCE_OUTPUT_DIMENSION

    @property
    def loss(self):
        return self._confidence_loss.loss

    @property
    def loss_name(self):
        return self._confidence_loss.name

    @property
    def loss_weight(self):
        return self._confidence_loss.weight

    @property
    def num_chunk_frames(self):
        return self._num_chunk_frames


class ConfidenceAuxTrainerHead(TrainerHeadInterface):

    def __init__(
            self, confidence_aux_predictor_head: "ConfidenceAuxPredictorHead",
            weight_creators: List[WeightCreatorInterface]) -> None:
        self._target_radii = confidence_aux_predictor_head.target_radii
        self._num_chunk_frames = confidence_aux_predictor_head.num_chunk_frames
        self._num_classes = confidence_aux_predictor_head.num_classes
        self._confidence_aux_predictor_head = confidence_aux_predictor_head
        self._weight_creators = weight_creators

    def video_targets(
            self, video_labels_from_task: LabelsFromTaskDict) -> np.ndarray:
        video_labels = video_labels_from_task[Task.SPOTTING][INDEX_LABELS]
        non_zeros = np.nonzero(video_labels)
        shape = video_labels.shape
        targets = _create_confidence_aux_targets(
            non_zeros, self._target_radii, shape)
        weight_inputs = create_multidimensional_video_weight_inputs(
            self._weight_creators, video_labels, targets)
        return np.concatenate([targets, weight_inputs], axis=2)

    def tf_chunk_targets_mapper(
            self, video_targets_and_weight_inputs, task: Task):
        valid_task = ConfidenceAuxTrainerHead._valid_task(task)
        return get_tf_multidimensional_targets_and_weights_mapper(
            video_targets_and_weight_inputs, self._num_chunk_frames,
            self._weight_creators, valid_task)

    def chunk_targets(
            self, video_targets_and_weight_inputs, start: int,
            mask: np.ndarray, task: Task) -> np.ndarray:
        valid_task = ConfidenceAuxTrainerHead._valid_task(task)
        return get_multidimensional_chunk_targets_and_weights(
            video_targets_and_weight_inputs, start, mask,
            self._num_chunk_frames, self._weight_creators, valid_task)

    @property
    def video_targets_shape(self):
        return (
            None, self._num_classes,
            len(self._target_radii) * CONFIDENCE_TARGET_DIMENSION
        )

    @property
    def predictor_head(self):
        return self._confidence_aux_predictor_head

    @property
    def supports_mixup(self) -> bool:
        return True

    @staticmethod
    def _valid_task(task: Task) -> bool:
        return task == Task.SPOTTING


class ConfidenceAuxPredictorHead(PredictorHeadInterface):

    def __init__(
            self, name: str, target_radii: List[float], num_chunk_frames:
            int, num_classes: int, confidence_aux_loss: "ConfidenceAuxLoss",
            weight_decay: float, batch_norm: bool, dropout_rate: float,
            width: int, num_head_layers: int) -> None:
        self._name = name
        self._target_radii = target_radii
        self._num_chunk_frames = num_chunk_frames
        self._num_classes = num_classes
        self._confidence_aux_loss = confidence_aux_loss
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._dropout_rate = dropout_rate
        self._width = width
        self._num_head_layers = num_head_layers

    def create_tensor(self, nodes: Nodes) -> Tensor:
        confidence_tensor_input, last_tensor_window_size = create_head_stack(
            self._name, self._weight_decay, self._batch_norm,
            self._dropout_rate, self._width, self._num_head_layers,
            nodes[NODE_PENULTIMATE])
        return _create_confidence_tensor(
            self._name, self._num_chunk_frames, self._num_classes,
            len(self._target_radii), self._weight_decay, self._batch_norm,
            self._dropout_rate, last_tensor_window_size,
            confidence_tensor_input, None)

    def post_process(self, confidences):
        # First dimension is for batch instances.
        return expit(confidences[0])

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dimension(self):
        return CONFIDENCE_OUTPUT_DIMENSION

    @property
    def loss(self):
        return self._confidence_aux_loss.loss

    @property
    def loss_name(self):
        return self._confidence_aux_loss.name

    @property
    def loss_weight(self):
        return self._confidence_aux_loss.weight

    @property
    def num_chunk_frames(self):
        return self._num_chunk_frames

    @property
    def target_radii(self):
        return self._target_radii


class ConfidenceLoss:

    def __init__(self, weight: float, focusing_gamma: float) -> None:

        # Note: don't change this function's name, since the name is
        # currently used when loading the models.
        def confidence_loss(targets_and_weights, predictions):
            targets = targets_and_weights[:, :, :, 0:1]
            weights = targets_and_weights[:, :, :, 1]
            return _create_confidence_loss(
                targets, predictions, weights, focusing_gamma)

        self.loss = confidence_loss
        self.name = "confidence_loss"
        self.weight = weight


class ConfidenceAuxLoss:

    def __init__(
            self, weight: float, focusing_gamma: float, n_radii: int) -> None:
        # Note: don't change this function's name, since the name is
        # currently used when loading the models.
        def confidence_aux_loss(targets_and_weights, predictions):
            losses = []
            for target_index in range(n_radii):
                current_targets = \
                    targets_and_weights[:, :, :, target_index:target_index + 1]
                current_weights = \
                    targets_and_weights[:, :, :, n_radii + target_index]
                current_predictions = \
                    predictions[:, :, :, target_index:target_index + 1]
                losses.append(_create_confidence_loss(
                    current_targets, current_predictions, current_weights,
                    focusing_gamma))
            return tf.add_n(losses) / n_radii

        self.loss = confidence_aux_loss
        self.name = "confidence_aux_loss"
        self.weight = weight


class ConfidenceClassCounts:

    def __init__(self, dataset: Dataset, target_radius: float) -> None:
        num_classes = dataset.num_classes_from_task[Task.SPOTTING]
        positive_counts_per_class = np.zeros(num_classes)
        negative_counts_per_class = np.zeros(num_classes)
        for video_datum in dataset.video_data:
            video_labels = video_datum.labels(Task.SPOTTING)
            video_positive_counts, video_negative_counts = \
                ConfidenceClassCounts._video_counts(
                    video_labels, target_radius)
            positive_counts_per_class += video_positive_counts
            negative_counts_per_class += video_negative_counts
        self.positive_counts_per_class = positive_counts_per_class
        self.negative_counts_per_class = negative_counts_per_class

    @staticmethod
    def _video_counts(
            video_labels: np.ndarray,
            target_radius: float) -> Tuple[np.ndarray, np.ndarray]:
        non_zeros = np.nonzero(video_labels)
        shape = video_labels.shape
        video_confidence_targets = _create_confidence_targets(
            non_zeros, target_radius, shape)
        video_positive_counts_per_class = np.sum(
            video_confidence_targets != NEGATIVE_CONFIDENCE, axis=0)
        video_negative_counts_per_class = (
                len(video_confidence_targets) - video_positive_counts_per_class)
        return video_positive_counts_per_class, video_negative_counts_per_class


class ConfidenceWeightCreator(WeightCreatorInterface):

    def __init__(
            self, confidence_class_counts: ConfidenceClassCounts,
            positive_weight: float, class_weights: np.ndarray) -> None:
        positive_weight_per_class, negative_weight_per_class = \
            compute_weights_from_counts(
                confidence_class_counts.positive_counts_per_class,
                confidence_class_counts.negative_counts_per_class,
                positive_weight)
        # Element-wise multiplication to take into account the class_weights.
        self._positive_weight_per_class = (
                class_weights * positive_weight_per_class)
        self._negative_weight_per_class = (
                class_weights * negative_weight_per_class)
        logging.info("Confidence positive_weight_per_class")
        logging.info(self._positive_weight_per_class)
        logging.info("Confidence negative_weight_per_class")
        logging.info(self._negative_weight_per_class)

    def video_weight_inputs(self, video_labels, video_targets):
        return np.where(
            video_targets != NEGATIVE_CONFIDENCE,
            self._positive_weight_per_class, self._negative_weight_per_class)

    def tf_chunk_weights(self, chunk_weight_inputs):
        return chunk_weight_inputs

    def chunk_weights(self, chunk_weight_inputs):
        return chunk_weight_inputs


def _create_confidence_loss(
        targets, logit_predictions, weights, focusing_gamma: float) -> Tensor:
    if focusing_gamma < 0.0:
        raise ValueError(
            "Value of gamma should be greater than or equal to zero.")
    bce = BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
    bce_losses = bce(targets, logit_predictions)
    # Note that we do not use the alpha here to weigh the different classes.
    # Weighing the classes should optionally be done via the input "weights"
    # parameter.
    weighted_losses = weights * bce_losses
    if focusing_gamma > 0.0:
        # Implement the focusing term of the focal loss, presented by Lin et al.
        # https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
        # p_error below corresponds to 1.0 - p_t in the focal loss paper.
        predictions = Activation("sigmoid")(logit_predictions)
        p_error = tf.squeeze(tf.math.abs(targets - predictions), axis=3)
        # When p_error is zero, the pow becomes problematic, leading to nan.
        # I'm not sure why.
        p_error_safe = tf.math.maximum(p_error, K.epsilon())
        modulating_factor = tf.pow(p_error_safe, focusing_gamma)
        weighted_losses = modulating_factor * weighted_losses
    return K.mean(weighted_losses, axis=(1, 2))


def _create_confidence_tensor(
        name, num_frames, num_classes, num_radii, weight_decay, batch_norm,
        dropout_rate: float, window_size: int, penultimate: Tensor,
        confidence_class_counts: Optional[ConfidenceClassCounts]) -> Tensor:
    if confidence_class_counts:
        # References:
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        # https://arxiv.org/pdf/1909.04868.pdf
        initial_biases = np.log(
            confidence_class_counts.positive_counts_per_class /
            confidence_class_counts.negative_counts_per_class)
        bias_initializer = tf.constant_initializer(initial_biases)
    else:
        bias_initializer = None
    convolution_name = f"{name}_{NODE_SUFFIX_CONVOLUTION}"
    confidences_convolution = convolution_wrapper(
        penultimate, num_classes * num_radii, (window_size, 1), (1, 1), "same",
        weight_decay, batch_norm, dropout_rate, INITIALIZER_FOR_SIGMOID,
        name=convolution_name, bias_initializer=bias_initializer)
    reshape = Reshape(
        (num_frames, num_classes, num_radii),
        name=f"{name}_{NODE_SUFFIX_LOGITS}")
    return reshape(confidences_convolution)


def _create_confidence_aux_targets(non_zeros, radii, shape):
    multiple_targets = [
        _create_confidence_targets(non_zeros, radius, shape)
        for radius in radii]
    return np.stack(multiple_targets, axis=2)


def _create_confidence_targets(non_zeros, radius, shape):
    frame_indexes, class_indexes = non_zeros
    n_frames, n_classes = shape
    confidences = np.zeros((n_frames, n_classes), dtype='float')
    for frame_index, class_index in zip(frame_indexes, class_indexes):
        # Set the confidence values to 1.0 around the positive example.
        positive_range = create_frame_range(frame_index, radius, n_frames)
        confidences[positive_range, class_index] = 1.0
    return confidences
