# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging
import math
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from spivak.data.dataset import VideoDatum, LabelsFromTaskDict, Task, \
    INDEX_LABELS, INDEX_VALID

SECONDS_IN_A_MINUTE = 60.0
MIN_VALID_CHUNK_FRAMES_WARNING_MULTIPLIER = 2.0

TFDataset = tf.data.Dataset


class VideoStartProviderInterface(metaclass=ABCMeta):

    @abstractmethod
    def create_tf_starts(
            self, video_labels_from_task: LabelsFromTaskDict) -> tf.Tensor:
        pass

    @abstractmethod
    def get_num_chunks_dataset_float(
            self, video_data: List[VideoDatum]) -> float:
        """Iterate over labels for all videos and accumulate the number of
        chunks that will be produced for them."""
        pass


class VideoStartProviderUniform(VideoStartProviderInterface):

    def __init__(
            self, chunks_per_minute: float, min_valid_chunk_frames: int,
            frame_rate: float, task: Task) -> None:
        self._chunks_per_minute = chunks_per_minute
        self._min_valid_chunk_frames = min_valid_chunk_frames
        self._frame_rate = frame_rate
        self._task = task

    def get_num_chunks_dataset_float(
            self, video_data: List[VideoDatum]) -> float:
        num_chunks_float_per_video = [
            _compute_num_chunks_float(
                self._frame_rate, video_datum.num_frames,
                self._chunks_per_minute)
            for video_datum in video_data
            if video_datum.valid_labels(self._task)
        ]
        return sum(num_chunks_float_per_video)

    def create_tf_starts(
            self, video_labels_from_task: LabelsFromTaskDict) -> tf.Tensor:
        video_labels = video_labels_from_task[self._task]
        valid_labels = video_labels[INDEX_VALID]
        num_video_frames = tf.shape(video_labels[INDEX_LABELS])[0]
        num_chunks = tf.cast(valid_labels, tf.int32) * _tf_sample_num_chunks(
            self._frame_rate, num_video_frames, self._chunks_per_minute)
        # random.uniform chooses integers from minval (0 by default) up to
        # maxval - 1, so we add 1 to start_end below.
        start_end = tf.maximum(
            0, num_video_frames - self._min_valid_chunk_frames) + 1
        return tf.random.uniform(
            shape=[num_chunks], maxval=start_end, dtype=tf.int32)


class VideoStartProviderWeighted(VideoStartProviderInterface):

    def __init__(
            self, chunks_per_minute: float, frame_rate: float,
            start_probabilities_creator: "StartProbabilitiesCreator") -> None:
        self._chunks_per_minute = chunks_per_minute
        self._frame_rate = frame_rate
        self._start_probabilities_creator = start_probabilities_creator

    def get_num_chunks_dataset_float(
            self, video_data: List[VideoDatum]) -> float:
        return sum(
            _compute_num_chunks_float(
                self._frame_rate, video_datum.num_frames,
                self._chunks_per_minute)
            for video_datum in video_data
        )

    def create_tf_starts(
            self, video_labels_from_task: LabelsFromTaskDict) -> tf.Tensor:
        return self._tf_get_starts(
            self._tf_get_start_probabilities(video_labels_from_task)
        )

    def _tf_get_start_probabilities(
            self, video_labels_from_task: LabelsFromTaskDict):
        # This function was designed to only be used with the spotting labels.
        video_labels = video_labels_from_task[Task.SPOTTING]
        video_start_probabilities = tf.py_function(
            func=self._video_start_probabilities, inp=[video_labels],
            Tout=tf.float32)
        # Can't return the Categorical distribution here, since the dataset
        # doesn't support it (I think it only supports tensors).
        return video_start_probabilities

    def _video_start_probabilities(self, video_labels):
        return self._start_probabilities_creator.create(video_labels)

    def _tf_get_starts(self, video_start_probabilities) -> tf.Tensor:
        num_video_frames = tf.shape(video_start_probabilities)[0]
        num_chunks = _tf_sample_num_chunks(
            self._frame_rate, num_video_frames, self._chunks_per_minute)
        distribution = tfp.distributions.Categorical(
            probs=video_start_probabilities)
        return distribution.sample(num_chunks)


class StartProbabilitiesCreator:

    def __init__(
            self, num_chunk_frames: int, min_valid_chunk_frames: int,
            negative_fraction: float) -> None:
        self._num_chunk_frames = num_chunk_frames
        self._min_valid_chunk_frames = min_valid_chunk_frames
        self._negative_fraction = negative_fraction

    def create(self, video_labels: np.ndarray):
        return _create_start_probabilities(
            video_labels, self._num_chunk_frames, self._min_valid_chunk_frames,
            self._negative_fraction)


def compute_min_valid_chunk_frames(
        video_data: List[VideoDatum], frame_rate: float,
        min_valid_chunk_duration: int):
    min_valid_chunk_frames = math.floor(frame_rate * min_valid_chunk_duration)
    _check_min_valid_chunk_frames(video_data, min_valid_chunk_frames)
    return min_valid_chunk_frames


def _check_min_valid_chunk_frames(
        video_data: List[VideoDatum], min_valid_chunk_frames: int) -> None:
    min_num_video_frames = min(
        video_datum.num_frames for video_datum in video_data)
    if (min_num_video_frames <
            MIN_VALID_CHUNK_FRAMES_WARNING_MULTIPLIER * min_valid_chunk_frames):
        logging.error(
            f"Dataset contains a video with {min_num_video_frames} frames. "
            f"The minimum frames set for a training chunk is "
            f"{min_valid_chunk_frames}. Consider decreasing this minimum if "
            f"there is a possibility of having a video smaller than that "
            f"in the test set.")


def _compute_num_chunks_float(
        frame_rate: float, num_frames: int, chunks_per_minute: float) -> float:
    num_seconds = num_frames / frame_rate
    chunks_per_second = chunks_per_minute / SECONDS_IN_A_MINUTE
    return num_seconds * chunks_per_second


def _tf_sample_num_chunks(frame_rate, n_video_frames, chunks_per_minute):
    num_video_frames_float = tf.cast(n_video_frames, tf.float32)
    num_seconds = num_video_frames_float / frame_rate
    chunks_per_second = chunks_per_minute / SECONDS_IN_A_MINUTE
    num_chunks_float = num_seconds * chunks_per_second
    # On average, we would like to produce num_chunks_float chunks.
    whole = tf.math.floor(num_chunks_float)
    fraction = num_chunks_float - whole
    additional = tf.less(tf.random.uniform(shape=[]), fraction)
    return tf.cast(whole, dtype=tf.int32) + tf.cast(additional, dtype=tf.int32)


def _create_start_probabilities(
        labels, num_chunk_frames: int, min_valid_chunk_frames: int,
        negative_fraction: float):
    # In general, for unbalanced classes, there is some benefit in doing
    # oversampling of minority classes. (See for example, "A systematic
    # study of the class imbalance problem in convolutional neural
    # networks", https://arxiv.org/pdf/1710.05381.pdf) Oversampling the
    # minority class can lead to better results and faster convergence,
    # though with an increased risk of over-fitting (see the above paper for
    # further information). For our problem, it's hard to know the right balance
    # between positive and negative examples, given that (for each chunk)
    # we create dense outputs, where windows of varying sizes around ground
    # truth samples may be considered as positive. We thus use a
    # fraction/bias determined empirically in order to sample negative
    # chunks. Unfortunately this adds yet another hyperparameter to the
    # code.
    num_frames = labels.shape[0]
    num_valid_starts = num_frames - min_valid_chunk_frames + 1
    # Important: we would like start_probabilities (and start_weights) to have
    # a length of num_frames, so that we can read and use the length later on,
    # when we compute the number of chunks to sample from each video. The
    # ending of start_probabilities will be filled with zeros usually,
    # since we don't want to sample starts from that region.
    start_weights = np.zeros(num_frames)
    # Sum across all classes, to determine how many labels are in each frame.
    label_counts = np.sum(labels == 1, axis=1)
    # Get the frames with at least one label.
    labeled_frames = np.nonzero(label_counts)[0]
    for labeled_frame in labeled_frames:
        # For each labeled frame, add weights to start_weights within a window
        # where a chunk that contains the frame in its center (accounting for
        # slack from the border due to the receptive field) may start.
        begin = min(
            num_valid_starts - 1,
            max(0, labeled_frame - num_chunk_frames + 1)
        )
        end = min(
            num_valid_starts, max(1, labeled_frame + 1))
        span_size = end - begin
        label_count = label_counts[labeled_frame]
        start_weights[begin:end] += label_count / span_size
    # Compute total weight assigned above to the regions near labels.
    positive_weight = np.sum(start_weights)
    if positive_weight == 0:
        # Here, since there are no weights in start_weights, negative_weight
        # can be arbitrary. The weights don't have to be normalized.
        negative_weight = 1.0
    else:
        positive_fraction = 1.0 - negative_fraction
        negative_weight = (positive_weight * negative_fraction /
                           positive_fraction)
    # If a position i has start_weights[i] == 0, it is not close to any labels,
    # so we consider that position as "negative", meaning it will produce
    # negative chunks (chunks with no labels close to its center).
    position_is_negative = (start_weights == 0)
    # We don't want to sample too far from the end, so that each chunk has a
    # minimum of min_valid_chunk_frames valid chunk frames.
    position_is_negative[num_valid_starts:] = 0
    negative_size = sum(position_is_negative)
    if negative_size > 0:
        # Negative regions should have a total weight of negative_weight,
        # to be spread out among all its positions.
        negative_frame_weight = negative_weight / negative_size
        # Assign the negative frame weight to negative positions.
        start_weights[position_is_negative] = negative_frame_weight
    # Normalize the weights in order to get a probability distribution.
    start_probabilities = start_weights / np.sum(start_weights)
    return start_probabilities
