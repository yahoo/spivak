# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from abc import ABCMeta, abstractmethod
from enum import IntEnum
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np

INDEX_LABELS = 0
INDEX_VALID = 1


class Task(IntEnum):
    SPOTTING = 0
    SEGMENTATION = 1


TASK_NAMES = {
    Task.SPOTTING: "spotting",
    Task.SEGMENTATION: "segmentation",
}

# Using only Dict and Tuple here in order to be able to manipulate
# LabelsFromTaskDict with TensorFlow's tensorflow.data library.
LabelsAndValid = Tuple[np.ndarray, bool]
LabelsFromTaskDict = Dict[Task, LabelsAndValid]
InputShape = Tuple[int, int, int]


class Dataset:

    def __init__(
            self, video_data: List["VideoDatum"], input_shape: InputShape,
            num_classes_from_task: Dict[Task, int]) -> None:
        self.video_data = video_data
        self.input_shape = input_shape
        self.tasks = list(num_classes_from_task.keys())
        self.num_classes_from_task = num_classes_from_task
        self.num_features = input_shape[1]
        self.num_videos = len(video_data)


class VideoDatum(metaclass=ABCMeta):

    @abstractmethod
    def labels(self, task: Task) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def valid_labels(self, task: Task) -> bool:
        pass

    @property
    @abstractmethod
    def labels_from_task(self) -> LabelsFromTaskDict:
        pass

    @property
    @abstractmethod
    def num_classes_from_task(self) -> Dict[Task, int]:
        pass

    @property
    @abstractmethod
    def features(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def relative_path(self) -> Path:
        pass

    @property
    @abstractmethod
    def num_features(self) -> int:
        pass

    @property
    @abstractmethod
    def num_frames(self) -> int:
        pass

    @property
    @abstractmethod
    def tasks(self) -> List[Task]:
        pass


class DefaultVideoDatum(VideoDatum):

    """This caches the labels, but not the features. This should work well
    for most use-cases."""

    def __init__(
            self, features_path: Path, relative_path: Path,
            labels_from_task: LabelsFromTaskDict, num_frames: int) -> None:
        self._features_path = features_path
        self._relative_path = relative_path
        self._labels_from_task = labels_from_task
        self._num_classes_from_task = {
            task: task_labels[INDEX_LABELS].shape[1]
            for task, task_labels in labels_from_task.items()}
        self._num_frames = num_frames

    def labels(self, task: Task) -> Optional[np.ndarray]:
        if task not in self._labels_from_task:
            return None
        return self._labels_from_task[task][INDEX_LABELS]

    def valid_labels(self, task: Task) -> bool:
        if task not in self._labels_from_task:
            return False
        return self._labels_from_task[task][INDEX_VALID]

    @property
    def labels_from_task(self) -> LabelsFromTaskDict:
        return self._labels_from_task

    @property
    def num_classes_from_task(self) -> Dict[Task, int]:
        return self._num_classes_from_task

    @property
    def features(self) -> np.ndarray:
        return np.load(str(self._features_path))

    @property
    def relative_path(self) -> Path:
        return self._relative_path

    @property
    def num_features(self) -> int:
        return _read_num_features(self._features_path)

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def tasks(self) -> List[Task]:
        return list(self._labels_from_task.keys())


def read_num_frames(features_path: Path) -> int:
    shape = read_numpy_shape(features_path)
    return shape[0]


def read_numpy_shape(features_path: Path) -> List[int]:
    with features_path.open('rb') as features_file:
        file_version = np.lib.format.read_magic(features_file)
        assert file_version == (1, 0)
        shape, _, _ = np.lib.format.read_array_header_1_0(features_file)
    return shape


def _read_num_features(features_path: Path) -> int:
    shape = read_numpy_shape(features_path)
    return shape[1]
