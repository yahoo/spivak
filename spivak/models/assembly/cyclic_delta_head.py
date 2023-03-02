# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Reshape

from spivak.models.assembly.delta_head import DeltaPredictorHeadInterface, \
    DELTA_OUTPUT_DIMENSION
from spivak.models.assembly.head import create_head_stack
from spivak.models.assembly.layers import Nodes, NODE_PENULTIMATE, \
    convolution_wrapper, NODE_SUFFIX_CONVOLUTION, INITIALIZER_FOR_SIGMOID

COSINE_EPSILON = 0.01


class CyclicDeltaPredictorHead(DeltaPredictorHeadInterface):

    def __init__(
            self, name: str, radius: float, delta_loss: "CyclicDeltaLoss",
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
        return create_cyclic_delta_tensor(
            self._name, self._num_chunk_frames, self._num_classes, 1,
            self._weight_decay, self._batch_norm, self._dropout_rate,
            delta_tensor_window_size, self._zero_init, delta_tensor_input)

    def post_process(self, delta_sin_cos):
        return post_process_cyclic_delta(delta_sin_cos, self._radius)

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


class CyclicDeltaLoss:

    def __init__(self, weight, base_loss):
        # Note: don't change this function's name, since the name is
        # currently used when loading the models. Should eventually fix this
        # issue.
        def cyclic_delta_loss(targets_and_weights, predictions):
            targets = targets_and_weights[:, :, :, 0:1]
            weights = targets_and_weights[:, :, :, 1]
            return create_cyclic_delta_loss(
                targets, predictions, weights, base_loss)

        self.weight = weight
        self.loss = cyclic_delta_loss
        self.name = "cyclic_delta_loss"


def create_cyclic_delta_tensor(
        name, num_chunk_frames, num_classes, num_radii, weight_decay,
        batch_norm, dropout_rate, window_size, zero_init: bool,
        penultimate: Tensor) -> Tensor:
    if zero_init:
        kernel_initializer = tf.keras.initializers.Zeros()
        cosine_bias_initializer = tf.keras.initializers.Constant(1.0)
    else:
        kernel_initializer = INITIALIZER_FOR_SIGMOID
        cosine_bias_initializer = None
    convolution_name = f"{name}_{NODE_SUFFIX_CONVOLUTION}"
    deltas_sine_convolution = convolution_wrapper(
        penultimate, num_classes * num_radii, (window_size, 1), (1, 1),
        "same", weight_decay, batch_norm, dropout_rate, kernel_initializer,
        name=f"{convolution_name}_sine", activation="tanh")
    deltas_cosine_convolution = convolution_wrapper(
        penultimate, num_classes * num_radii, (window_size, 1), (1, 1),
        "same", weight_decay, batch_norm, dropout_rate, kernel_initializer,
        bias_initializer=cosine_bias_initializer,
        name=f"{convolution_name}_cosine", activation="tanh")
    # stack does not take in a name argument.
    deltas_convolution = K.stack(
        [deltas_sine_convolution, deltas_cosine_convolution], axis=4)
    reshape = Reshape((num_chunk_frames, num_classes, num_radii * 2), name=name)
    return reshape(deltas_convolution)


def post_process_cyclic_delta(delta_sin_cos, radius):
    sine = delta_sin_cos[0, :, :, 0:1]
    cosine = delta_sin_cos[0, :, :, 1:2]
    angles_with_nans = np.arctan2(sine, cosine)
    # Replace nans with zeros.
    angles = np.nan_to_num(angles_with_nans, nan=0.0)
    # Convert from [-pi, pi] to [-radius, radius]
    return (radius / np.pi) * angles


def create_cyclic_delta_loss(targets, predictions, weights, base_loss):
    return (
        _create_cyclic_delta_diff_loss(targets, predictions, weights, base_loss)
        + 0.01 * _unit_norm_loss(predictions, weights)
    )


def _unit_norm_loss(y_pred, weights):
    return K.mean(weights * tf.abs(1.0 - tf.norm(y_pred, axis=3)), axis=(1, 2))


def _create_cyclic_delta_diff_loss(
        targets, sin_cos_predictions, weights, base_loss):
    clipped_targets = tf.clip_by_value(targets, -1.0, 1.0)
    targets_angle = np.pi * clipped_targets
    targets_sin = tf.sin(targets_angle)
    targets_cos = tf.cos(targets_angle)
    targets_sin_cos = tf.concat([targets_sin, targets_cos], axis=3)
    # Following suggestion here:
    # https://scicomp.stackexchange.com/questions/27689/numerically-stable-way-of-computing-angles-between-vectors

    # We get nan from atan2 when the norm of the sum is close to zero. This
    # should happen when prediction and target are diametrically opposed,
    # which often happens during the beginning of the optimization, when the
    # models is producing random outputs. We do clip the norm of the sum
    # below, so that it is not too close to zero. This doesn't matter
    # much in terms of modifying the objective, as the norm of the sum will
    # eventually be large after some training, so the clipping won't be
    # affecting the results.
    normalized_predictions = tf.math.l2_normalize(sin_cos_predictions, axis=3)
    norm_diff = tf.norm(targets_sin_cos - normalized_predictions, axis=3)
    norm_sum = tf.norm(targets_sin_cos + normalized_predictions, axis=3)
    norm_sum_clipped = tf.maximum(norm_sum, COSINE_EPSILON)

    # Note that since both parameters of atan2 are positive,
    # they always represent an angle in the first quadrant. This makes the
    # optimization behave reasonably well. When the angle difference is close
    # to 0, which will happen often, atan2 and its derivative are well defined.
    angle_diff = 2 * tf.atan2(norm_diff, norm_sum_clipped)
    # The largest possible angle diff is pi, due to the wrapping around. We
    # convert the range from [0, pi] to [0, 1].
    diff = angle_diff / np.pi
    diff = tf.expand_dims(diff, axis=3)
    all_losses = base_loss(diff, 0.0)
    return K.mean(weights * all_losses, axis=(1, 2))
