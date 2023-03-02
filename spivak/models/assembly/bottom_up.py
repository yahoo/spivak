# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from abc import ABCMeta, abstractmethod
from typing import List

from tensorflow import Tensor
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D

from spivak.models.assembly.convolution_stacks import StridedBlockInterface, \
    ConvolutionStackInterface

POOLING_MAX = "max"
POOLING_AVERAGE = "average"


class BottomUpStackInterface(metaclass=ABCMeta):

    @abstractmethod
    def downsample_and_convolve(
            self, bottom_up: Tensor, num_filters_in: int, num_filters_out: int,
            layer_index: int) -> Tensor:
        pass

    @abstractmethod
    def convolve(
            self, bottom_up: Tensor, num_filters: int,
            layer_index: int) -> Tensor:
        pass


class BottomUpLayer:

    def __init__(self, tensor: Tensor, num_channels: int) -> None:
        self.tensor = tensor
        self.num_channels = num_channels


class PoolingBottomUpStack(BottomUpStackInterface):

    def __init__(
            self, pooling: str,
            convolution_stack: ConvolutionStackInterface) -> None:
        self._pooling = pooling
        self._convolution_stack = convolution_stack
        self._name = "bu"

    def convolve(
            self, bottom_up: Tensor, num_filters: int,
            layer_index: int) -> Tensor:
        return self._convolution_stack.convolve(
            bottom_up, num_filters, layer_index, self._name)

    def downsample_and_convolve(
            self, bottom_up: Tensor, num_filters_in: int, num_filters_out: int,
            layer_index: int) -> Tensor:
        if self._pooling == POOLING_MAX:
            pooling = MaxPooling2D(
                (2, 1), name=f"{self._name}_layer{layer_index}_max_pooling")
        elif self._pooling == POOLING_AVERAGE:
            pooling = AveragePooling2D(
                (2, 1), name=f"{self._name}_layer{layer_index}_average_pooling")
        else:
            raise ValueError(f"Unrecognized pooling: {self._pooling}")
        pooled = pooling(bottom_up)
        return self._convolution_stack.convolve(
            pooled, num_filters_out, layer_index, self._name)


class StridedBottomUpStack(BottomUpStackInterface):

    def __init__(
            self, strided_block: StridedBlockInterface,
            layer_num_blocks: List[int], strided_reduction: bool) -> None:
        self._strided_block = strided_block
        self._layer_num_blocks = layer_num_blocks
        self._strided_reduction = strided_reduction
        self._name = "bu"

    def convolve(
            self, bottom_up: Tensor, num_filters: int,
            layer_index: int) -> Tensor:
        for block_index in range(self._layer_num_blocks[layer_index]):
            bottom_up = self._strided_block.convolve(
                bottom_up, num_filters,
                name=f"{self._name}_layer{layer_index}_block{block_index}")
        return bottom_up

    def downsample_and_convolve(
            self, bottom_up: Tensor, num_filters_in: int, num_filters_out: int,
            layer_index: int) -> Tensor:
        num_blocks_in_layer = self._layer_num_blocks[layer_index]
        if self._strided_reduction or num_blocks_in_layer < 2:
            stride_filters = num_filters_out
        else:
            stride_filters = num_filters_in
        bottom_up = self._strided_block.strided_convolve(
            bottom_up, stride_filters,
            name=f"{self._name}_layer{layer_index}_block0")
        for block_index in range(1, num_blocks_in_layer):
            bottom_up = self._strided_block.convolve(
                bottom_up, num_filters_out,
                name=f"{self._name}_layer{layer_index}_block{block_index}")
        return bottom_up


def create_bottom_up_layers(
        input_mlp_out: Tensor, num_layers: int, base_num_filters: int,
        max_num_filters: int, bottom_up_stack: BottomUpStackInterface
) -> List[BottomUpLayer]:
    # VGG-16 applied dropout of 0.5 to their last two layers. U-net paper did
    # something similar, only applying dropout at the end of their bottom-up
    # layers.
    # https://arxiv.org/pdf/1409.1556.pdf
    # https://arxiv.org/pdf/1505.04597.pdf
    bottom_up_layers = []
    # Start with just a convolution stack and add that to the layers.
    num_filters_out = min(max_num_filters, base_num_filters)
    x = bottom_up_stack.convolve(input_mlp_out, num_filters_out, 0)
    bottom_up_layers.append(BottomUpLayer(x, num_filters_out))
    # Now, for each layer, add a stack that downsamples then convolves.
    for layer_index in range(1, num_layers):
        num_filters_in = num_filters_out
        num_filters_out = min(
            max_num_filters, 2 ** layer_index * base_num_filters)
        x = bottom_up_stack.downsample_and_convolve(
            x, num_filters_in, num_filters_out, layer_index)
        bottom_up_layers.append(BottomUpLayer(x, num_filters_out))
    return bottom_up_layers
