# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from typing import List, Tuple

import numpy as np
from tensorflow.python.keras import Model

from spivak.data.video_chunk_iterator import VideoChunkIteratorProvider


class Projector:

    def __init__(
            self, projector_model: Model,
            video_chunk_iterator_provider: VideoChunkIteratorProvider) -> None:
        self._projector_model = projector_model
        self._video_chunk_iterator_provider = video_chunk_iterator_provider

    def project(self, video_features: np.ndarray) -> np.ndarray:
        input_chunk_batch, valid_chunk_sizes = self._prepare_input_batch(
            video_features)
        chunk_output_batch = self._projector_model.predict(input_chunk_batch)
        valid_chunk_outputs = [
            chunk_output_batch[c][0:valid_chunk_size]
            for c, valid_chunk_size in enumerate(valid_chunk_sizes)]
        return self._accumulate_outputs(valid_chunk_outputs, video_features)

    def _prepare_input_batch(
            self, features: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        input_chunk_iterator = self._video_chunk_iterator_provider.provide(
            features)
        return input_chunk_iterator.prepare_input_batch()

    def _accumulate_outputs(
            self, valid_chunk_outputs: List[np.ndarray],
            features: np.ndarray) -> np.ndarray:
        chunk_output_iterator = self._video_chunk_iterator_provider.provide(
            features)
        num_frames = features.shape[0]
        num_projected_features = valid_chunk_outputs[0].shape[2]
        projected_features = np.zeros((num_frames, 1, num_projected_features))
        chunk_output_iterator.accumulate_chunk_outputs(
            projected_features, valid_chunk_outputs)
        return np.squeeze(projected_features, axis=1)
