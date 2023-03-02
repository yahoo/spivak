# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from typing import Tuple, List

import numpy as np


class VideoChunkIteratorProvider:
    """This is to be used to help run inference over test videos and
    aggregate the results."""

    def __init__(self, chunk_frames: int, num_border_frames: int) -> None:
        self._chunk_frames = chunk_frames
        self._num_border_frames = num_border_frames

    def provide(self, video_features: np.ndarray) -> "VideoChunkIterator":
        return VideoChunkIterator(
            video_features, self._chunk_frames, self._num_border_frames)


class VideoChunkIterator:

    def __init__(
            self, video_features: np.ndarray, chunk_frames: int,
            num_border_frames: int) -> None:
        self.chunk_features_expanded = None
        self.valid_chunk_size = None
        self._output_start = None
        self._output_end = None
        self._result_start = None
        self._result_end = None
        self._is_last = False
        self._chunk_start = 0
        self._chunk_frames = chunk_frames
        self._num_border_frames = num_border_frames
        self._video_features_expanded = np.expand_dims(video_features, axis=-1)
        self._num_frames = self._video_features_expanded.shape[0]

    def prepare_input_batch(self) -> Tuple[np.ndarray, List[int]]:
        input_chunk_batch_list = []
        valid_chunk_sizes = []
        while self.has_next():
            self.next()
            input_chunk_batch_list.append(self.chunk_features_expanded)
            valid_chunk_sizes.append(self.valid_chunk_size)
        return np.concatenate(input_chunk_batch_list), valid_chunk_sizes

    def accumulate_chunk_outputs(
            self, accumulated_output: np.ndarray,
            output_chunks: List[np.ndarray]) -> None:
        chunk_index = 0
        while self.has_next():
            self.next()
            self.accumulate(accumulated_output, output_chunks[chunk_index])
            chunk_index += 1

    def has_next(self) -> bool:
        return not self._is_last

    def next(self) -> None:
        # Get the outputs for this chunk and store the results.
        self.valid_chunk_size = min(
            self._chunk_frames,
            self._video_features_expanded.shape[0] - self._chunk_start)
        chunk_end = self._chunk_start + self.valid_chunk_size
        if self.valid_chunk_size == self._chunk_frames:
            # This might be faster than creating the np.zeros as below,
            # but not sure.
            chunk_features_expanded = \
                self._video_features_expanded[self._chunk_start:chunk_end]
        else:
            # in this case, data_expanded is not as big as chunk_size, so we add
            # extra zero-padding before passing it through the network.
            chunk_features_expanded = np.zeros((
                self._chunk_frames, self._video_features_expanded.shape[1],
                self._video_features_expanded.shape[2]))
            chunk_features_expanded[0:self.valid_chunk_size] = \
                self._video_features_expanded[self._chunk_start:chunk_end]
        # Prepare the batch made of one chunk for the network
        self.chunk_features_expanded = np.expand_dims(
            chunk_features_expanded, axis=0)
        # Figure out start and end indexes.
        is_first = (self._chunk_start == 0)
        if is_first:
            self._output_start = 0
        else:
            self._output_start = self._num_border_frames
        self._result_start = self._chunk_start + self._output_start
        self._is_last = (
                self._chunk_start >= self._num_frames - self._chunk_frames)
        if self._is_last:
            self._output_end = self.valid_chunk_size
        else:
            self._output_end = (self.valid_chunk_size -
                                self._num_border_frames)
        self._result_end = self._chunk_start + self._output_end
        # Update the start index for the next iteration
        self._chunk_start += self._chunk_frames - 2 * self._num_border_frames
        if self._chunk_start > self._num_frames - self._chunk_frames:
            self._chunk_start = self._num_frames - self._chunk_frames

    def accumulate(
            self, accumulated_output: np.ndarray,
            output_chunk: np.ndarray) -> None:
        result_start = self._result_start
        result_end = self._result_end
        output_start = self._output_start
        output_end = self._output_end
        accumulated_output[result_start:result_end] = \
            output_chunk[output_start:output_end]
