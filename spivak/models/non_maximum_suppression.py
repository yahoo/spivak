# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import csv
import math
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from spivak.data.label_map import LabelMap

MIN_DETECTION_SCORE_SUPPRESS = 0.0
# This was tweaked by looking at the results and the size of the resulting
# JSON files. In the experiments, 1e-3 was already enough to give good final
# scores, so 1e-5 is a bit of overkill, but JSON file sizes are still
# manageable (around 2 or 3MB).
MIN_DETECTION_SCORE_LINEAR = 1e-5


class FlexibleNonMaximumSuppression:

    NMS_COLUMN_WINDOW = "window"
    NMS_COLUMN_LABEL = "label"

    def __init__(
            self, nms_on: bool, class_windows: Optional[np.ndarray],
            score_decay: "ScoreDecayInterface") -> None:
        """class_windows are not in seconds, but in frames."""
        self._nms_on = nms_on
        self._class_windows = class_windows
        self._score_decay = score_decay

    def maybe_apply(self, detection_scores: np.ndarray) -> np.ndarray:
        if not self._nms_on:
            return detection_scores
        return _flexible_non_maximum_suppression(
            detection_scores, self._class_windows, self._score_decay)

    @staticmethod
    def read_nms_windows(windows_path: Path, label_map: LabelMap) -> np.ndarray:
        num_classes = label_map.num_classes()
        class_windows_in_seconds = np.empty(num_classes)
        # Set everything to nan, so we can later make sure all values were set.
        class_windows_in_seconds[:] = np.nan
        with windows_path.open("r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                label = row[FlexibleNonMaximumSuppression.NMS_COLUMN_LABEL]
                # Just ignore the label if it's not in the label map.
                if label in label_map.label_to_int:
                    label_int = label_map.label_to_int[label]
                    class_windows_in_seconds[label_int] = row[
                        FlexibleNonMaximumSuppression.NMS_COLUMN_WINDOW]
        # Make sure all values have been filled in.
        assert not np.any(np.isnan(class_windows_in_seconds))
        return class_windows_in_seconds


class ScoreDecayInterface:

    @abstractmethod
    def decay(self, scores: np.ndarray, max_index: int, window: float) -> None:
        """Decay (or suppress) scores in a window around the max value."""
        pass

    @property
    @abstractmethod
    def min_detection_score(self) -> float:
        pass


class ScoreDecaySuppress(ScoreDecayInterface):

    def decay(self, scores: np.ndarray, max_index: int, window: float) -> None:
        """This is standard non-maximum suppression, where a score gets
        completely suppressed if it is in the neighborhood of the max value."""
        int_radius = math.floor(window / 2.0)
        start = max(max_index - int_radius, 0)
        end = min(max_index + int_radius + 1, scores.shape[0])
        scores[start:end] = -1

    @property
    def min_detection_score(self) -> float:
        return MIN_DETECTION_SCORE_SUPPRESS


class ScoreDecayLinear(ScoreDecayInterface):

    def __init__(self, min_weight: float, window_expansion: float) -> None:
        self._min_weight = min_weight
        self._window_expansion = window_expansion

    def decay(self, scores: np.ndarray, max_index: int, window: float) -> None:
        # So that the effect of the linear decay is more comparable to
        # that of the regular NMS (suppression done by ScoreDecaySuppress),
        # we expand the window used for the linear decay here.
        expanded_window = self._window_expansion * window
        radius = expanded_window / 2.0
        radius_ceil = math.ceil(radius)
        start = max(max_index - radius_ceil, 0)
        end = min(max_index + radius_ceil + 1, scores.shape[0])
        frame_range = np.arange(start, end)
        weights = self._min_weight + (1.0 - self._min_weight) * np.abs(
            (frame_range - max_index) / radius)
        clipped_weights = np.clip(weights, 0.0, 1.0)
        scores[frame_range] *= clipped_weights
        # Remove the max_index.
        scores[max_index] = -1

    @property
    def min_detection_score(self) -> float:
        return MIN_DETECTION_SCORE_LINEAR


def _flexible_non_maximum_suppression(
        scores: np.ndarray, class_windows: np.ndarray,
        score_decay: ScoreDecayInterface) -> np.ndarray:
    # Apply nms separately for each class.
    nms_scores_list = [
        _single_class_nms(
            scores[:, class_index], class_window, score_decay)
        for class_index, class_window in enumerate(class_windows)
    ]
    return np.column_stack(nms_scores_list)


def _single_class_nms(
        class_scores: np.ndarray, class_window: float,
        score_decay: ScoreDecayInterface) -> np.ndarray:
    class_scores_nms = np.zeros(class_scores.shape) - 1
    class_scores_tmp = np.copy(class_scores)
    max_index = int(np.argmax(class_scores_tmp))
    max_score = class_scores_tmp[max_index]
    min_detection_score = score_decay.min_detection_score
    while max_score >= min_detection_score:
        # Copy the highest value over to the final result.
        class_scores_nms[max_index] = max_score
        # Suppress the scores in class_scores_tmp
        score_decay.decay(class_scores_tmp, max_index, class_window)
        # Find the maximum from the remaining values.
        max_index = np.argmax(class_scores_tmp)
        max_score = class_scores_tmp[max_index]
    return class_scores_nms
