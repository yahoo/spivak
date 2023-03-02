# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from spivak.data.dataset import LabelsFromTaskDict, Task, INDEX_LABELS
from spivak.models.assembly.cyclic_delta_head import \
    post_process_cyclic_delta, create_cyclic_delta_loss, \
    create_cyclic_delta_tensor
from spivak.models.assembly.delta_head import DELTA_TARGET_DIMENSION, \
    DELTA_OUTPUT_DIMENSION, create_delta_targets
from spivak.models.assembly.head import TrainerHeadInterface, \
    create_multidimensional_video_weight_inputs, \
    get_tf_multidimensional_targets_and_weights_mapper, \
    get_multidimensional_chunk_targets_and_weights, PredictorHeadInterface, \
    create_head_stack
from spivak.models.assembly.layers import Nodes, NODE_PENULTIMATE
from spivak.models.assembly.weight_creator import WeightCreatorInterface


class DeltaAuxTrainerHead(TrainerHeadInterface):

    def __init__(
            self, delta_aux_predictor_head: "DeltaAuxPredictorHead",
            weight_creators: List[WeightCreatorInterface]) -> None:
        self._radii = delta_aux_predictor_head.radii
        self._num_chunk_frames = delta_aux_predictor_head.num_chunk_frames
        self._num_classes = delta_aux_predictor_head.num_classes
        self._delta_aux_predictor_head = delta_aux_predictor_head
        self._weight_creators = weight_creators

    def video_targets(
            self, video_labels_from_task: LabelsFromTaskDict) -> np.ndarray:
        video_labels = video_labels_from_task[Task.SPOTTING][INDEX_LABELS]
        non_zeros = np.nonzero(video_labels)
        shape = video_labels.shape
        targets = _create_delta_aux_targets(non_zeros, self._radii, shape)
        weight_inputs = create_multidimensional_video_weight_inputs(
            self._weight_creators, video_labels, targets)
        return np.concatenate([targets, weight_inputs], axis=2)

    def tf_chunk_targets_mapper(
            self, video_targets_and_weight_inputs, task: Task):
        valid_task = DeltaAuxTrainerHead._valid_task(task)
        return get_tf_multidimensional_targets_and_weights_mapper(
            video_targets_and_weight_inputs, self._num_chunk_frames,
            self._weight_creators, valid_task)

    def chunk_targets(
            self, video_targets_and_weight_inputs, start: int,
            mask: np.ndarray, task: Task) -> np.ndarray:
        valid_task = DeltaAuxTrainerHead._valid_task(task)
        return get_multidimensional_chunk_targets_and_weights(
            video_targets_and_weight_inputs, start, mask,
            self._num_chunk_frames, self._weight_creators, valid_task)

    @property
    def video_targets_shape(self):
        return (
            None, self._num_classes, len(self._radii) * DELTA_TARGET_DIMENSION)

    @property
    def predictor_head(self):
        return self._delta_aux_predictor_head

    @property
    def supports_mixup(self) -> bool:
        return False

    @staticmethod
    def _valid_task(task: Task) -> bool:
        return task == Task.SPOTTING


class DeltaAuxPredictorHead(PredictorHeadInterface):

    def __init__(
            self, name: str, radii: List[float],
            delta_aux_loss: "CyclicDeltaAuxLoss", num_chunk_frames: int,
            num_classes: int, weight_decay: float, batch_norm: bool,
            dropout_rate: float, width: int, num_head_layers: int,
            zero_init: bool) -> None:
        self._name = name
        self._radii = radii
        self._delta_aux_loss = delta_aux_loss
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
        return create_cyclic_delta_tensor(
            self._name, self._num_chunk_frames, self._num_classes,
            len(self._radii), self._weight_decay, self._batch_norm,
            self._dropout_rate, delta_tensor_window_size,
            self._zero_init, delta_tensor_input)

    def post_process(self, delta_sin_cos):
        # TODO: run for all radii
        return post_process_cyclic_delta(
            delta_sin_cos[:, :, :, 0:2], self._radii[0])

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dimension(self):
        # TODO: Fix post_process and multiply below by len(self._radii)
        return DELTA_OUTPUT_DIMENSION

    @property
    def loss(self):
        return self._delta_aux_loss.loss

    @property
    def loss_name(self):
        return self._delta_aux_loss.name

    @property
    def loss_weight(self):
        return self._delta_aux_loss.weight

    @property
    def num_chunk_frames(self):
        return self._num_chunk_frames

    @property
    def radii(self):
        return self._radii


class CyclicDeltaAuxLoss:

    def __init__(self, weight, base_loss, n_radii):
        # Note: don't change this function's name, since the name is
        # currently used when loading the models. It would be good to eventually
        # fix this naming constraint.
        def cyclic_delta_aux_loss(targets_and_weights, predictions):
            losses = []
            for target_index in range(n_radii):
                prediction_index = 2 * target_index
                current_targets = \
                    targets_and_weights[:, :, :, target_index:target_index + 1]
                current_weights = \
                    targets_and_weights[:, :, :, n_radii + target_index]
                current_predictions = \
                    predictions[:, :, :, prediction_index:prediction_index + 2]
                losses.append(create_cyclic_delta_loss(
                    current_targets, current_predictions, current_weights,
                    base_loss))
            return tf.add_n(losses) / n_radii

        self.weight = weight
        self.loss = cyclic_delta_aux_loss
        self.name = "cyclic_delta_aux_loss"


def _create_delta_aux_targets(non_zeros, radii: List[float], shape):
    multiple_targets = [
        create_delta_targets(non_zeros, radius, shape) for radius in radii]
    return np.stack(multiple_targets, axis=2)
