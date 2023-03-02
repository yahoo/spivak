# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from typing import Optional, Dict, Tuple, Union

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Input
from tensorflow.keras import backend
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, \
    Dropout

from spivak.data.dataset import InputShape

backend.set_image_data_format('channels_last')

Nodes = Dict[str, Tensor]

NODE_PENULTIMATE = "penultimate"
NODE_MAIN_INPUT = "main_input"
NODE_OPTIONAL_PROJECTION = "optional_projection"
NODE_INPUT_REDUCTION_1 = "input_reduction_1"
NODE_INPUT_REDUCTION_OUT = "input_reduction_out"
NODE_SUFFIX_CONVOLUTION = "convolution"
NODE_SUFFIX_HEAD_STACK_CONVOLUTION = "head_stack_convolution"
NODE_SUFFIX_BATCH_NORMALIZATION = "batch_normalization"
NODE_SUFFIX_DROPOUT = "dropout"
NODE_SUFFIX_LOGITS = "logits"

# For some reason, glorot_uniform gives better results than he_uniform and
# he_normal, even when used with relu.
INITIALIZER_FOR_RELU = "glorot_uniform"  # "he_uniform"
INITIALIZER_FOR_SIGMOID = "glorot_uniform"
INITIALIZER_FOR_SOFTMAX = "glorot_uniform"

TARGET_PROJECTED_DIMENSION = 512


def create_main_input(input_shape: InputShape) -> Tensor:
    return Input(shape=input_shape, dtype='float32', name=NODE_MAIN_INPUT)


def input_reduction(
        main_input: Tensor, num_features: int, num_out_channels: int,
        weight_decay: float, batch_norm: int, dropout_rate: float) -> Tensor:
    # Add an optional linear layer that reduces the feature dimensionality to
    # TARGET_PROJECTED_DIMENSION, in case it starts out larger than that.
    if num_features > TARGET_PROJECTED_DIMENSION:
        optional_projection = convolution_wrapper(
            main_input, TARGET_PROJECTED_DIMENSION, (1, num_features), (1, 1),
            "valid", weight_decay, batch_norm, dropout_rate,
            INITIALIZER_FOR_RELU, name=NODE_OPTIONAL_PROJECTION,
            activation="relu")
        second_window_size = 1
    else:
        optional_projection = main_input
        second_window_size = num_features
    # The next two lines implement a 2-layer MLP to reduce the dimensionality
    # of feature vectors at each frame. Normally, the dimensionality will go
    # from 512 (TARGET_PROJECTED_DIMENSION) to 32 (4 * width). The two layers
    # below in effect implement 1x1 convolutions, in that each frame is
    # processed independently from the others. The first parameter to
    # convolution_2d() is the number of filters (output channels), while the
    # second is the shape of each filter.
    input_reduction_1 = convolution_wrapper(
        optional_projection, 4 * num_out_channels, (1, second_window_size),
        (1, 1), "valid", weight_decay, batch_norm, dropout_rate,
        INITIALIZER_FOR_RELU, name=NODE_INPUT_REDUCTION_1, activation="relu")
    input_reduction_out = convolution_wrapper(
        input_reduction_1, num_out_channels, (1, 1), (1, 1), "valid",
        weight_decay, batch_norm, dropout_rate, INITIALIZER_FOR_RELU,
        name=NODE_INPUT_REDUCTION_OUT, activation="relu")
    return input_reduction_out


def conv_2d(
        num_filters: int, kernel_size: Tuple[int, int],
        strides: Tuple[int, int], padding: str, weight_decay: float,
        kernel_initializer: Union[str, Initializer], name: Optional[str] = None,
        use_bias: Optional[bool] = True,
        bias_initializer: Optional[Initializer] = None) -> Conv2D:
    if not bias_initializer:
        # "zeros" is the default value used by Keras.
        bias_initializer = "zeros"
    if weight_decay:
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
    return Conv2D(
        num_filters, kernel_size, strides=strides, padding=padding,
        use_bias=use_bias, bias_initializer=bias_initializer,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer, name=name)


def convolution_wrapper(
        tensor: Tensor, num_filters: int, kernel_size: Tuple[int, int],
        strides: Tuple[int, int], padding: str, weight_decay: float,
        batch_norm: int, dropout_rate: float,
        kernel_initializer: Union[str, Initializer],
        name: Optional[str] = None, activation: Optional[str] = None,
        pre_activated: Optional[bool] = False, use_bias: Optional[bool] = True,
        bias_initializer: Optional[Initializer] = None) -> Tensor:
    # Don't use both weight decay and batch-norm at the same time, according to
    # https://arxiv.org/pdf/1706.05350.pdf
    assert not (weight_decay and batch_norm), (
        "Make sure you really want to combine weight decay and batch"
        "normalization")
    if pre_activated:
        tensor = optional_activation(tensor, activation, name)
        post_activation = None
    else:
        post_activation = activation
    convolution = conv_2d(
        num_filters, kernel_size, strides, padding, weight_decay,
        kernel_initializer, name, use_bias, bias_initializer)
    return generic_convolution_block(
        tensor, convolution, batch_norm, dropout_rate, name, post_activation)


def generic_convolution_block(
        input_tensor: Tensor, convolution: Conv2D, batch_norm: int,
        dropout_rate: float, name: Optional[str],
        activation: Optional[str]) -> Tensor:
    # Try to avoid using both dropout and batch-norm at the same time,
    # according to https://arxiv.org/pdf/1801.05134.pdf
    assert not (dropout_rate and batch_norm), (
        "Try to avoid combining dropout and batch normalization")
    convolution_output = convolution(input_tensor)
    convolution_output = optional_activation(
        convolution_output, activation, name)
    # There are results suggesting that batch-norm works well when put after
    # the activation function:
    # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
    if batch_norm:
        if name:
            batch_norm_name = f"{name}_{NODE_SUFFIX_BATCH_NORMALIZATION}"
        else:
            batch_norm_name = None
        # Channel-wise batch normalization.
        convolution_output = BatchNormalization(
            axis=3, name=batch_norm_name)(convolution_output)
    # This seems like a reasonable place for the dropout. See for example
    # this work: https://arxiv.org/pdf/1904.03392.pdf
    if dropout_rate:
        if name:
            dropout_name = f"{name}_{NODE_SUFFIX_DROPOUT}"
        else:
            dropout_name = None
        convolution_output = Dropout(
            rate=dropout_rate, name=dropout_name)(convolution_output)
    return convolution_output


def optional_activation(
        tensor: Tensor, activation: Optional[str], name: Optional[str]
) -> Tensor:
    if not activation:
        return tensor
    # My version of Keras doesn't work with sigmoid activation inside
    # Conv2D, and we want to be able to do dropout before the activation
    # anyway, so adding the activation here instead of in the Conv2D call.
    if name:
        activation_name = f"{name}_{activation}"
    else:
        activation_name = None
    return Activation(activation, name=activation_name)(tensor)


def kernel_initializer_from_activation(activation: Optional[str]) -> str:
    if activation == "relu":
        return INITIALIZER_FOR_RELU
    return INITIALIZER_FOR_SIGMOID
