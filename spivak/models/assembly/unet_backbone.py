# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

# Relevant papers:
# U-net paper: https://arxiv.org/pdf/1505.04597.pdf
# FCN is somewhat similar to UNet, but simpler and seems to be worse:
# https://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf
# LinkNet, which is also pretty similar to u-net, but more recent, and seems to
# be a bit better.
# https://arxiv.org/pdf/1707.03718.pdf

from abc import ABCMeta, abstractmethod
from typing import List, Optional

from tensorflow import Tensor
from tensorflow.keras.layers import Concatenate, UpSampling2D, Activation, \
    Conv2DTranspose, Add

from spivak.models.assembly.bottom_up import BottomUpLayer
from spivak.models.assembly.convolution_stacks import ConvolutionStackInterface
from spivak.models.assembly.layers import Nodes, NODE_PENULTIMATE, \
    convolution_wrapper, generic_convolution_block, optional_activation, \
    kernel_initializer_from_activation, INITIALIZER_FOR_RELU


class UpsamplerInterface(metaclass=ABCMeta):

    @abstractmethod
    def upsample(self, top_down: Tensor, num_filters: int, name: str) -> Tensor:
        pass


class CombinerInterface(metaclass=ABCMeta):

    @abstractmethod
    def combine(
            self, bottom_up: Tensor, top_down: Tensor, num_filters: int,
            name: str) -> Tensor:
        pass


class TopDownStackInterface(metaclass=ABCMeta):

    @abstractmethod
    def expand_and_combine(
            self, bottom_up: Tensor, top_down: Tensor, num_filters: int,
            layer_index: int) -> Tensor:
        pass

    @abstractmethod
    def expand(
            self, top_down: Tensor, num_filters: int,
            layer_index: int) -> Tensor:
        pass

    @abstractmethod
    def optional_post_activation(self, top_down: Tensor) -> Tensor:
        pass


class ConvTransposeUpsampler(UpsamplerInterface):

    def __init__(
            self, kernel_size, padding: str, weight_decay: float,
            batch_norm: int, dropout_rate: float, activation: str,
            pre_activated: bool) -> None:
        self._kernel_size = kernel_size
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._dropout_rate = dropout_rate
        self._padding = padding
        self._activation = activation
        self._pre_activated = pre_activated

    def upsample(
            self, top_down: Tensor, num_filters: int, name: str) -> Tensor:
        conv_transpose_name = f"{name}_transpose"
        if self._pre_activated:
            top_down = optional_activation(
                top_down, self._activation, conv_transpose_name)
            post_activation = None
        else:
            post_activation = self._activation
        conv_2d_transpose = Conv2DTranspose(
            num_filters, kernel_size=self._kernel_size, strides=(2, 1),
            padding=self._padding, kernel_initializer=INITIALIZER_FOR_RELU,
            name=conv_transpose_name)
        return generic_convolution_block(
            top_down, conv_2d_transpose, self._batch_norm,
            self._dropout_rate, conv_transpose_name, post_activation)


class UpsamplingUpsampler(UpsamplerInterface):

    def __init__(
            self, interpolation: str, convolve: bool, kernel_size, padding: str,
            weight_decay: float, batch_norm: int, dropout_rate: float,
            activation: str, pre_activated: bool) -> None:
        self._interpolation = interpolation
        self._convolve = convolve
        self._kernel_size = kernel_size
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._dropout_rate = dropout_rate
        self._padding = padding
        self._activation = activation
        self._kernel_initializer = kernel_initializer_from_activation(
            activation)
        self._pre_activated = pre_activated

    def upsample(self, top_down: Tensor, num_filters: int, name: str) -> Tensor:
        upsampling_name = f"{name}_upsampling"
        # I'm assuming it doesn't make much sense to put an activation around
        # the upsampling, but we have it later in the convolution that follows.
        upsampling = UpSampling2D(
            size=(2, 1), interpolation=self._interpolation,
            name=upsampling_name)
        upsampled = upsampling(top_down)
        if not self._convolve:
            return upsampled
        return convolution_wrapper(
            upsampled, num_filters, self._kernel_size, (1, 1), self._padding,
            self._weight_decay, self._batch_norm, self._dropout_rate,
            self._kernel_initializer, upsampling_name, self._activation,
            self._pre_activated)


class ConcatenationCombiner(CombinerInterface):

    def combine(
            self, bottom_up: Tensor, top_down: Tensor, num_filters: int,
            name: str) -> Tensor:
        concatenate_operation = Concatenate(
            axis=-1, name=f"{name}_combine_concatenate")
        return concatenate_operation([bottom_up, top_down])


class AdditionCombiner(CombinerInterface):
    """
    It's not clear if it's better to use addition or concatenation,
    and whether to use a 1x1 convolution before the addition or not. Some
    papers that use addition are mentioned here for reference.
    See LinkNet:
    https://arxiv.org/pdf/1707.03718.pdf
    LinkNet is a u-net-like architecture. They simply use addition in the
    skip connections, without 1x1 convolutions.
    See also 2015 FCN paper, Section 4.2:
    https://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf
    They also have a u-net-like architecture. They do the lateral connections
    using a 1x1 convolution in order to do have the result be able to do class
    predictions. Then they then add the layers together.
    See also Figure 3 in FPN paper at:
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf
    FPN uses different number of channels in the encoder and decoder,
    which justifies their use of 1x1 convolutions when doing the skip
    connections, which also uses addition.
    """

    def __init__(
            self, pre_convolve: bool, weight_decay: float, batch_norm: int,
            dropout_rate: float, activation: str, pre_activated: bool) -> None:
        self._pre_convolve = pre_convolve
        self._weight_decay = weight_decay
        self._batch_norm = batch_norm
        self._dropout_rate = dropout_rate
        self._activation = activation
        self._kernel_initializer = kernel_initializer_from_activation(
            activation)
        self._pre_activated = pre_activated

    def combine(
            self, bottom_up: Tensor, top_down: Tensor, num_filters: int,
            name: str) -> Tensor:
        if self._pre_convolve:
            bottom_up = convolution_wrapper(
                bottom_up, num_filters, (1, 1), (1, 1), "same",
                self._weight_decay, self._batch_norm, self._dropout_rate,
                self._kernel_initializer, f"{name}_combine_conv",
                self._activation, self._pre_activated)
        addition_operation = Add(name=f"{name}_combine_add")
        return addition_operation([bottom_up, top_down])


class IgnoreCombiner(CombinerInterface):

    def combine(
            self, bottom_up: Tensor, top_down: Tensor, num_filters: int,
            name: str) -> Tensor:
        """Ignores bottom-up contribution and simply returns the top-down
        input. To be used for ablation purposes."""
        return top_down


class BasicTopDownStack(TopDownStackInterface):

    def __init__(
            self, upsampler: UpsamplerInterface, combiner: CombinerInterface,
            convolution_stack: Optional[ConvolutionStackInterface],
            pre_activated: bool) -> None:
        self._upsampler = upsampler
        self._combiner = combiner
        self._convolution_stack = convolution_stack
        self._pre_activated = pre_activated
        self._name = "td"

    def expand_and_combine(
            self, bottom_up: Tensor, top_down: Tensor, num_filters: int,
            layer_index: int) -> Tensor:
        # Upsample the top_down features by 2X.
        upsampled = self._upsampler.upsample(
            top_down, num_filters, f"{self._name}_layer{layer_index}")
        # Combine the bottom-up features with the upsampled top-down ones,
        # either by addition or concatenation.
        combined = self._combiner.combine(
            bottom_up, upsampled, num_filters,
            f"{self._name}_layer{layer_index}")
        if not self._convolution_stack:
            return combined
        return self._convolution_stack.convolve(
            combined, num_filters, layer_index, self._name)

    def expand(
            self, top_down: Tensor, num_filters: int,
            layer_index: int) -> Tensor:
        upsampled = self._upsampler.upsample(
            top_down, num_filters, f"{self._name}_layer{layer_index}")
        return self._convolution_stack.convolve(
            upsampled, num_filters, layer_index, self._name)

    def optional_post_activation(self, top_down: Tensor) -> Tensor:
        if not self._pre_activated:
            return top_down
        return Activation("relu", name=f"{self._name}_post_relu")(top_down)


def create_unet_backbone(
        bottom_up_layers: List[BottomUpLayer], layers_start: int,
        layers_end: int, top_down_stack: TopDownStackInterface) -> Nodes:
    selected_layer_indexes = list(range(layers_start, layers_end))
    top_down = bottom_up_layers[selected_layer_indexes[-1]].tensor
    for layer_index in reversed(selected_layer_indexes[:-1]):
        bottom_up_layer = bottom_up_layers[layer_index]
        # Traditional u-net matches the number of filters in the bottom-up
        # and top-down layers, as does the linknet. FPN on the other hand,
        # directly predicts from their top-down layers using a shared head,
        # so they keep the number of top-down channels constant.
        # https://arxiv.org/pdf/1505.04597.pdf (u-net)
        # https://arxiv.org/pdf/1707.03718.pdf (linknet)
        # https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf
        # (FPN)
        num_filters = bottom_up_layer.num_channels
        top_down = top_down_stack.expand_and_combine(
            bottom_up_layer.tensor, top_down, num_filters, layer_index)
    # Need to upsample back to the original input tensor size when
    # bottom_up_layers_start is larger than 0.
    start_num_filters = bottom_up_layers[layers_start].num_channels
    for layer_index in reversed(range(layers_start)):
        num_filters = (
            start_num_filters //
            2 ** (layers_start - layer_index)
        )
        top_down = top_down_stack.expand(top_down, num_filters, layer_index)
    top_down = top_down_stack.optional_post_activation(top_down)
    return {NODE_PENULTIMATE: top_down}
