# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from tensorflow import Tensor
from tensorflow.keras import backend
from tensorflow.keras.layers import Activation, MaxPooling2D, Add, \
    ZeroPadding2D, Layer

from spivak.models.assembly.layers import convolution_wrapper, \
    kernel_initializer_from_activation, INITIALIZER_FOR_RELU


class ConvolutionStackInterface(metaclass=ABCMeta):

    @abstractmethod
    def convolve(
            self, tensor: Tensor, num_filters: int, layer_index: int,
            name: str) -> Tensor:
        pass


class ConvolutionBlockInterface(metaclass=ABCMeta):

    @abstractmethod
    def convolve(self, tensor: Tensor, num_filters: int, name: str) -> Tensor:
        pass


class StridedBlockInterface(ConvolutionBlockInterface):

    @abstractmethod
    def strided_convolve(
            self, tensor: Tensor, num_filters: int, name: str) -> Tensor:
        pass


class BasicConvolutionStack(ConvolutionStackInterface):

    def __init__(
            self, convolution_block: ConvolutionBlockInterface,
            layer_num_blocks: List[int]) -> None:
        self._convolution_block = convolution_block
        self._layer_num_blocks = layer_num_blocks

    def convolve(
            self, tensor: Tensor, num_filters: int, layer_index: int,
            name: str) -> Tensor:
        for block_index in range(self._layer_num_blocks[layer_index]):
            tensor = self._convolution_block.convolve(
                tensor, num_filters,
                f"{name}_layer{layer_index}_block{block_index}")
        return tensor


class BasicConvolutionBlock(ConvolutionBlockInterface):

    def __init__(
            self, padding: str, weight_decay: float, batch_norm: int,
            dropout_rate: float, activation: str) -> None:
        self._padding = padding
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._dropout_rate = dropout_rate
        self._activation = activation
        self._kernel_initializer = kernel_initializer_from_activation(
            activation)

    def convolve(self, tensor: Tensor, num_filters: int, name: str) -> Tensor:
        return convolution_wrapper(
            tensor, num_filters, (3, 1), (1, 1), self._padding,
            self._weight_decay, self._batch_norm, self._dropout_rate,
            self._kernel_initializer, name=name, activation=self._activation)


class Scaling(Layer):

    def __init__(self, **kwargs):
        super(Scaling, self).__init__(**kwargs)
        self._scale = backend.variable(0.0, dtype="float32")

    def call(self, inputs):
        return inputs * self._scale


class ResidualBlockV2(StridedBlockInterface):

    # Implementation of the second version of the residual block,
    # as published by He et al. in a follow-up paper to their original
    # residual networks paper: https://arxiv.org/pdf/1603.05027.pdf
    # The motivation for this implementation was to be able to use SkipInit,
    # which was published in https://arxiv.org/pdf/2002.10444.pdf

    def __init__(
            self, num_convolutions: int, skip_init: bool, bottleneck: bool,
            weight_decay: float, batch_norm: int, dropout_rate: float) -> None:
        if bottleneck:
            assert num_convolutions == 1
        self._num_convolutions = num_convolutions
        self._skip_init = skip_init
        self._bottleneck = bottleneck
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._dropout_rate = dropout_rate
        # Original ResNet V2 does not use bias in two of its three bottleneck
        # convolutions, whereas SkipInit paper uses bias in all three, stating
        # slightly better results.
        self._use_bias = True

    def convolve(
            self, tensor: Tensor, num_filters: int, name: str) -> Tensor:
        return self._residual_block(tensor, num_filters, (1, 1), name)

    def strided_convolve(
            self, tensor: Tensor, num_filters: int, name: str) -> Tensor:
        return self._residual_block(tensor, num_filters, (2, 1), name)

    def _residual_block(
            self, input_tensor: Tensor, num_filters: int,
            final_strides: Tuple[int, int], name: str) -> Tensor:
        input_tensor_relu = Activation(
            "relu", name=f"{name}_conv0_relu")(input_tensor)
        tensor = self._internal_convolutions(
            input_tensor_relu, num_filters, final_strides, name)
        if self._should_use_shortcut():
            shortcut = self._prepare_shortcut(
                input_tensor, input_tensor_relu, num_filters, final_strides,
                name)
            if self._skip_init:
                # Learnable multiplier, starting at zero to make learning safer.
                scaling = Scaling(name=f"{name}_residual_multiply")
                tensor = scaling(tensor)
            addition = Add(name=f"{name}_shortcut_add")
            tensor = addition([shortcut, tensor])
        return tensor

    def _internal_convolutions(
            self, tensor: Tensor, num_filters: int,
            final_strides: Tuple[int, int], name: str) -> Tensor:
        if self._bottleneck:
            return self._bottleneck_convolutions(
                tensor, num_filters, final_strides, name)
        else:
            return self._standard_convolutions(
                tensor, num_filters, final_strides, name)

    def _should_use_shortcut(self) -> bool:
        """Don't use a shortcut in the degenerate case when there is only one
        non-bottleneck convolution"""
        return self._bottleneck or self._num_convolutions > 1

    def _prepare_shortcut(
            self, input_tensor: Tensor, input_tensor_relu: Tensor,
            num_filters: int, strides: Tuple[int, int], name: str) -> Tensor:
        if num_filters != input_tensor.shape[3]:
            # 1x1 convolution to adjust num_channels (with num_filters) and
            # possibly also num_frames (via strides)
            return convolution_wrapper(
                input_tensor_relu, num_filters, (1, 1), strides, "same",
                self._weight_decay, self._batch_norm, self._dropout_rate,
                INITIALIZER_FOR_RELU, name=f"{name}_shortcut_conv")
        elif strides == (1, 1):
            # The input_tensor is already in the right shape, so it should be
            # used directly as the shortcut.
            return input_tensor
        else:
            # Do a stride without convolutions. Note that the pool size
            # below is (1, 1), so no pooling happens, just striding.
            pooling = MaxPooling2D((1, 1), strides, name=f"{name}_striding")
            return pooling(input_tensor)

    def _standard_convolutions(
            self, input_tensor_relu: Tensor, num_filters: int,
            final_strides: Tuple[int, int], name: str) -> Tensor:
        tensor = input_tensor_relu
        for convolution_index in range(0, self._num_convolutions):
            if convolution_index > 0:
                # Don't run activation on the first iteration,
                # as input_tensor_relu already comes after an activation.
                tensor = Activation(
                    "relu", name=f"{name}_conv{convolution_index}_relu")(tensor)
            if convolution_index == self._num_convolutions - 1:
                # Last convolution might be strided in order to down-sample
                # the frames.
                strides = final_strides
            else:
                strides = (1, 1)
            tensor = convolution_wrapper(
                tensor, num_filters, (3, 1), strides, "same",
                self._weight_decay, self._batch_norm, self._dropout_rate,
                INITIALIZER_FOR_RELU, name=f"{name}_conv{convolution_index}")
        return tensor

    def _bottleneck_convolutions(
            self, input_tensor_relu: Tensor, num_filters: int,
            final_strides: Tuple[int, int], name: str) -> Tensor:
        assert num_filters % 4 == 0
        bottleneck_num_filters = num_filters // 4
        # 1 x 1 convolution to reduce the number of channels. Note this is
        # not strided, as the stride comes only in the second convolution.
        # This is different than in the regular ResidualBlock (V1).
        tensor = convolution_wrapper(
            input_tensor_relu, bottleneck_num_filters, (1, 1), (1, 1),
            "valid", self._weight_decay, self._batch_norm, self._dropout_rate,
            INITIALIZER_FOR_RELU, name=f"{name}_reduce",
            use_bias=self._use_bias)
        tensor = Activation('relu', name=f"{name}_reduce_relu")(tensor)
        tensor = ZeroPadding2D(
            padding=((1, 1), (0, 0)), name=f"{name}_pad")(tensor)
        # 3 x 1 convolution with few channels. This convolution uses the
        # provided strides, so will possibly reduce the number of frames
        # (when strides > (1, 1)).
        tensor = convolution_wrapper(
            tensor, bottleneck_num_filters, (3, 1), final_strides,
            "valid", self._weight_decay, self._batch_norm, self._dropout_rate,
            INITIALIZER_FOR_RELU, name=f"{name}_convolve",
            use_bias=self._use_bias)
        tensor = Activation('relu', name=f"{name}_convolve_relu")(tensor)
        # 1 x 1 convolution to recover original number of channels. Unlike
        # the previous convolutions, this one should always use the bias.
        return convolution_wrapper(
            tensor, num_filters, (1, 1), (1, 1), "valid", self._weight_decay,
            self._batch_norm, self._dropout_rate, INITIALIZER_FOR_RELU,
            name=f"{name}_expand")


class ResidualBlock(StridedBlockInterface):

    def __init__(
            self, num_convolutions: int, bottleneck: bool, weight_decay: float,
            batch_norm: int, dropout_rate: float) -> None:
        if bottleneck:
            assert num_convolutions == 1
        self._num_convolutions = num_convolutions
        self._bottleneck = bottleneck
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._dropout_rate = dropout_rate

    def convolve(
            self, tensor: Tensor, num_filters: int, name: str) -> Tensor:
        return self._residual_block(tensor, num_filters, (1, 1), name)

    def strided_convolve(
            self, tensor: Tensor, num_filters: int, name: str) -> Tensor:
        return self._residual_block(tensor, num_filters, (2, 1), name)

    def _residual_block(
            self, input_tensor: Tensor, num_filters: int,
            initial_strides: Tuple[int, int], name: str) -> Tensor:
        tensor = self._internal_convolutions(
            input_tensor, num_filters, initial_strides, name)
        if self._should_use_shortcut():
            shortcut = self._prepare_shortcut(
                input_tensor, num_filters, initial_strides, name)
            addition = Add(name=f"{name}_shortcut_add")
            tensor = addition([shortcut, tensor])
        return Activation("relu", name=f"{name}_relu")(tensor)

    def _internal_convolutions(
            self, tensor: Tensor, num_filters: int,
            strides: Tuple[int, int], name: str) -> Tensor:
        if self._bottleneck:
            return self._bottleneck_convolutions(
                tensor, num_filters, strides, name)
        else:
            return self._standard_convolutions(
                tensor, num_filters, strides, name)

    def _should_use_shortcut(self) -> bool:
        """Don't use a shortcut in the degenerate case when there is only one
        non-bottleneck convolution"""
        return self._bottleneck or self._num_convolutions > 1

    def _prepare_shortcut(
            self, tensor: Tensor, num_filters: int,
            strides: Tuple[int, int], name: str) -> Tensor:
        if strides == (1, 1) and num_filters == tensor.shape[3]:
            # The input tensor is already in the right shape, so it should be
            # used directly as the shortcut.
            return tensor
        # 1x1 convolution to adjust num_frames (with strides)
        # and/or num_channels (with num_filters).
        return convolution_wrapper(
            tensor, num_filters, (1, 1), strides, "same", self._weight_decay,
            self._batch_norm, self._dropout_rate, INITIALIZER_FOR_RELU,
            name=f"{name}_shortcut_conv")

    def _standard_convolutions(
            self, tensor: Tensor, num_filters: int,
            initial_strides: Tuple[int, int], name: str) -> Tensor:
        for convolution_index in range(0, self._num_convolutions):
            if convolution_index == 0:
                # First convolution might be strided in order to down-sample.
                strides = initial_strides
            else:
                strides = (1, 1)
            if convolution_index == self._num_convolutions - 1:
                activation = None
            else:
                activation = "relu"
            tensor = convolution_wrapper(
                tensor, num_filters, (3, 1), strides, "same",
                self._weight_decay, self._batch_norm, self._dropout_rate,
                INITIALIZER_FOR_RELU, name=f"{name}_conv{convolution_index}",
                activation=activation)
        return tensor

    def _bottleneck_convolutions(
            self, tensor: Tensor, num_filters: int, strides: Tuple[int, int],
            name: str) -> Tensor:
        assert num_filters % 4 == 0
        bottleneck_num_filters = num_filters // 4
        # 1 x 1 convolution to reduce the number of channels. This uses
        # the provided strides, so will possibly reduce the number of
        # frames (when strides > (1, 1)).
        tensor = convolution_wrapper(
            tensor, bottleneck_num_filters, (1, 1), strides, "valid",
            self._weight_decay, self._batch_norm, self._dropout_rate,
            INITIALIZER_FOR_RELU, activation="relu", name=f"{name}_contract")
        # 3 x 1 convolution with few channels.
        tensor = convolution_wrapper(
            tensor, bottleneck_num_filters, (3, 1), (1, 1), "same",
            self._weight_decay, self._batch_norm, self._dropout_rate,
            INITIALIZER_FOR_RELU, activation="relu", name=f"{name}_convolve")
        # 1 x 1 convolution to recover original number of channels.
        return convolution_wrapper(
            tensor, num_filters, (1, 1), (1, 1), "valid",
            self._weight_decay, self._batch_norm, self._dropout_rate,
            INITIALIZER_FOR_RELU, name=f"{name}_expand")
