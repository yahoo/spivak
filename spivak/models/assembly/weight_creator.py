# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import csv
import math
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from spivak.data.label_map import LabelMap

CLASS_WEIGHTS_COLUMN_LABEL = "label"
CLASS_WEIGHTS_COLUMN_WEIGHT = "weight"
# EXPECTED_POSITIVE_RATE should be more or less the expected number of positive
# labels in a video divided by the number of frames in the video. I'm not
# sure that we'll really use this, as it seems the negative_sampling_rate
# parameter to create_frame_weights() works better when set to 1.0, which means
# we always end up using all samples anyway.
# For SoccerNet V1, there was around 1 action every seven minutes, while for
# SoccerNet V2, there were around 2 actions per minute.
# EXPECTED_POSITIVE_RATE = 1.0 / (7.0 * 60.0 * 2.0)  # V1
EXPECTED_POSITIVE_RATE = 2.0 / (1.0 * 60.0 * 2.0)  # V2


class WeightCreatorInterface(metaclass=ABCMeta):

    @abstractmethod
    def video_weight_inputs(self, video_labels, video_targets):
        pass

    @abstractmethod
    def tf_chunk_weights(self, chunk_weight_inputs):
        pass

    @abstractmethod
    def chunk_weights(self, chunk_weight_inputs):
        pass


class IdentityWeightCreator(WeightCreatorInterface):

    def __init__(self, class_weights: np.ndarray) -> None:
        self._class_weights = class_weights

    def video_weight_inputs(
            self, video_labels: np.ndarray,
            video_targets: np.ndarray) -> np.ndarray:
        num_frames = video_labels.shape[0]
        return np.tile(self._class_weights, (num_frames, 1))

    def tf_chunk_weights(self, chunk_weight_inputs):
        return chunk_weight_inputs

    def chunk_weights(self, chunk_weight_inputs):
        return chunk_weight_inputs


class SampledWeightCreator(WeightCreatorInterface):

    def __init__(
            self, weight_radius: float, negative_sampling_rate: float) -> None:
        # TODO: maybe add support for input class weights.
        self._weight_radius = weight_radius
        self._negative_sampling_rate = negative_sampling_rate
        expected_selection_rate = _compute_expected_selection_rate(
            self._weight_radius, self._negative_sampling_rate)
        self._normalizer = expected_selection_rate

    def video_weight_inputs(self, video_labels, video_targets):
        non_zeros = np.nonzero(video_labels)
        shape = video_labels.shape
        return _sampled_video_weight_inputs(
            non_zeros, self._weight_radius, shape)

    def tf_chunk_weights(self, chunk_weight_inputs):
        return _tf_sampled_chunk_weights(
            chunk_weight_inputs, self._negative_sampling_rate, self._normalizer)

    def chunk_weights(self, chunk_weight_inputs):
        return _sampled_chunk_weights(
            chunk_weight_inputs, self._negative_sampling_rate, self._normalizer)


def read_class_weights(
        class_weights_path: Path, label_map: LabelMap) -> np.ndarray:
    if not class_weights_path.exists():
        return np.ones(label_map.num_classes())
    num_classes = label_map.num_classes()
    class_weights = np.empty(num_classes)
    # Set everything to nan so we can later make sure all values were set.
    class_weights[:] = np.nan
    with class_weights_path.open("r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = row[CLASS_WEIGHTS_COLUMN_LABEL]
            label_int = label_map.label_to_int[label]
            class_weights[label_int] = row[CLASS_WEIGHTS_COLUMN_WEIGHT]
    # Make sure all values have been filled in.
    assert not np.any(np.isnan(class_weights))
    return class_weights


def create_frame_range(
        frame_index: int, radius: float, n_frames: int) -> np.ndarray:
    start = max(math.ceil(frame_index - radius), 0)
    end = min(math.floor(frame_index + radius) + 1, n_frames)
    return np.arange(start, end)


def _compute_expected_selection_rate(
        radius: float, negative_sampling_rate: float) -> float:
    # TODO: If we're going to actually use this code flow, compute it based on a
    #  set of videos, instead of trying to guess the positive_selection_rate,
    #  which does not account for overlaps.
    # We want the expected selection rate to be independent of
    # the size of the particular input videos, so that we don't reduce the
    # overall frame weights on larger videos. In other words, we would like
    # larger videos to have a larger influence over the weights.
    positive_rate = EXPECTED_POSITIVE_RATE
    positive_selection_rate = min(1.0, 2 * radius * positive_rate)
    return ((1.0 - positive_selection_rate) * negative_sampling_rate
            + positive_selection_rate * 1.0)


def _sampled_video_weight_inputs(non_zeros, radius: float, shape):
    num_frames, num_classes = shape
    weights = np.zeros((num_frames, num_classes))
    frame_indexes, class_indexes = non_zeros
    for frame_index, class_index in zip(frame_indexes, class_indexes):
        # Set the weights to in a window around the positive example.
        weight_range = create_frame_range(frame_index, radius, num_frames)
        weights[weight_range, class_index] = 1.0
    return weights


def _tf_sampled_chunk_weights(
        chunk_frame_weight_windows, negative_sampling_rate, normalizer):
    random_negative_frames = _tf_sample_negative_frames(
        negative_sampling_rate, tf.shape(chunk_frame_weight_windows))
    selected_frames = tf.maximum(
        chunk_frame_weight_windows, random_negative_frames)
    return selected_frames / normalizer


def _sampled_chunk_weights(
        frame_weight_windows: np.ndarray, negative_sampling_rate: float,
        normalizer: float):
    selected_frames = _sample_negative_frames(
        negative_sampling_rate, frame_weight_windows.shape)
    selected_frames[frame_weight_windows > 0] = 1.0
    return selected_frames / normalizer


def _tf_sample_negative_frames(negative_sampling_rate: float, shape):
    if negative_sampling_rate == 1.0:
        return tf.ones(shape)
    elif negative_sampling_rate == 0.0:
        return tf.zeros(shape)
    else:
        return tfp.distributions.Bernoulli(
            probs=negative_sampling_rate, dtype=tf.float32).sample(shape)


def _sample_negative_frames(negative_sampling_rate, shape):
    if negative_sampling_rate == 1.0:
        return np.ones(shape)
    elif negative_sampling_rate == 0.0:
        return np.zeros(shape)
    else:
        return np.random.binomial(n=1, p=negative_sampling_rate, size=shape)
