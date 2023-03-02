# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import json
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np

from spivak.data.dataset import Dataset, read_num_frames, DefaultVideoDatum, \
    Task, LabelsAndValid
from spivak.data.dataset_splits import name_from_path
from spivak.data.label_map import LabelMap

LABEL_FILE_KEY_LABEL = "label"
LABEL_FILE_KEY_TIME = "time"


class CustomDatasetReader:

    def __init__(
            self, video_label_readers: Dict[Task, "VideoLabelReader"],
            chunk_frames: int) -> None:
        self._video_label_readers = video_label_readers
        self._chunk_frames = chunk_frames

    def read(self, features_paths: List[Path],
             labels_path_dicts: List[Dict[Task, Path]]) -> Dataset:
        # Create relative paths, used later for writing results.
        relative_paths = [
            Path(name_from_path(features_path))
            for features_path in features_paths]
        num_frames_list = [
            read_num_frames(features_path) for features_path in features_paths]
        # Directly read video labels into memory, since they don't take up that
        # much space.
        labels_from_task_list = [
            {
                task: self._video_label_readers[task].read_video_labels(
                    labels_path_dict[task], num_frames)
                for task in labels_path_dict
            }
            for labels_path_dict, num_frames in zip(
                labels_path_dicts, num_frames_list)
        ]
        video_data = [
            DefaultVideoDatum(
                features_path, relative_path, labels_from_task, num_frames)
            for features_path, labels_from_task, relative_path, num_frames in
            zip(features_paths, labels_from_task_list, relative_paths,
                num_frames_list)
        ]
        num_features = video_data[0].num_features
        input_shape = (self._chunk_frames, num_features, 1)
        num_classes_from_task = {
            task: self._video_label_readers[task].num_classes
            for task in self._video_label_readers}
        return Dataset(video_data, input_shape, num_classes_from_task)


class VideoLabelReader:

    def __init__(self, label_file_reader: "LabelFileReaderInterface") -> None:
        self.num_classes: int = label_file_reader.num_classes
        self._label_file_reader = label_file_reader

    def read_video_labels(
            self, labels_path: Optional[Path],
            features_len: int) -> LabelsAndValid:
        if not labels_path:
            return self._empty_labels_and_valid(features_len)
        if not labels_path.exists():
            logging.warning(
                f"Could not find labels path, skipping labels: {labels_path}")
            return self._empty_labels_and_valid(features_len)
        return (
            self._label_file_reader.read_one_hot(labels_path, features_len),
            True)

    def _empty_labels_and_valid(self, length: int) -> LabelsAndValid:
        return np.zeros((length, self.num_classes)), False


class LabelFileReaderInterface(metaclass=ABCMeta):

    @abstractmethod
    def read_one_hot(self, json_path: Path, features_len: int) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass


class GenericLabelFileReader(LabelFileReaderInterface):

    EXCESS_SECONDS_FOR_WARNING = 1.0

    def __init__(
            self, label_map: LabelMap,
            label_groups: Optional[Dict[str, List[str]]],
            frame_rate: float) -> None:
        self._label_map = label_map
        self._label_groups = label_groups
        self._frame_rate = frame_rate
        self._num_classes = label_map.num_classes()

    def read_one_hot(self, json_path: Path, features_len: int) -> np.ndarray:
        with json_path.open() as json_file:
            json_data = json.load(json_file)
        raw_labels = np.zeros((features_len, self._num_classes))
        last_frame_index = features_len - 1
        for label_dict in json_data:
            time_in_seconds = label_dict[LABEL_FILE_KEY_TIME]
            frame = round(self._frame_rate * time_in_seconds)
            # Check if the current label is too far off from the end of the
            # video and give out a warning if so.
            frame_excess_in_seconds = (
                    (frame - last_frame_index) / self._frame_rate)
            # The last frames sometimes do not get decoded, so we allow for
            # some slack here.
            if (frame_excess_in_seconds >
                    GenericLabelFileReader.EXCESS_SECONDS_FOR_WARNING):
                video_approximate_duration = features_len / self._frame_rate
                logging.error(
                    f"Ignoring label with too large a time in json file "
                    f"{json_path.name}. The video has {features_len} frames "
                    f"(roughly {video_approximate_duration} s) while the label "
                    f"frame was {frame} ({time_in_seconds} s).")
            else:
                # Since we already verified that the frame is not too far off
                # from the end of the video, here we just push the frame
                # index into bounds.
                frame = min(frame, last_frame_index)
                derived_labels = self._derived_labels(
                    label_dict[LABEL_FILE_KEY_LABEL])
                for derived_label in derived_labels:
                    label_int = self._label_map.label_to_int[derived_label]
                    raw_labels[frame][label_int] = 1
        return raw_labels

    def _derived_labels(self, original_label: str) -> List[str]:
        potential_labels = [original_label]
        if self._label_groups and original_label in self._label_groups:
            group_labels = self._label_groups[original_label]
            potential_labels.extend(group_labels)
        valid_labels = [
            label for label in potential_labels
            if label in self._label_map.label_to_int]
        invalid_labels = [
            label for label in potential_labels
            if label not in self._label_map.label_to_int]
        for invalid_label in invalid_labels:
            logging.warning(
                f"Ignoring label {invalid_label}, which is not in the "
                f"label map.")
        return valid_labels

    @property
    def num_classes(self) -> int:
        return self._num_classes


def read_label_groups(
        label_groups_path: Path) -> Optional[Dict[str, List[str]]]:
    if not label_groups_path.exists():
        return None
    with label_groups_path.open("r") as groups_file:
        return json.load(groups_file)
