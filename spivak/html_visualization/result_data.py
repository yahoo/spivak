# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas
from pandas import DataFrame

from spivak.data.label_map import LabelMap

# Pandas dataframe columns
COLUMN_TIME = "Time"
COLUMN_SEGMENTATION_SCORE = "Segmentation score"
COLUMN_SEGMENTATION_LABEL = "Segmentation label"
COLUMN_CLASS = "Class"
COLUMN_DETECTION_SCORE = "Detection score"
COLUMN_DETECTION_SCORE_NMS = "Detection score after NMS"
COLUMN_CONFIDENCE = "Confidence"
COLUMN_DETECTION = "Detection"
COLUMN_SPOTTING_LABEL = "Spotting label"
COLUMN_DELTA = "Delta"
COLUMN_SOURCE = "Source"
COLUMN_SOURCE_FLOAT = "Source float"
# Filename suffixes
GENERIC_SUFFIX = ".npy"
DETECTION_SCORE_SUFFIX = "_detection_score.npy"
DETECTION_SCORE_NMS_SUFFIX = "_detection_score_nms.npy"
DETECTION_SUFFIX = "_detection_thresholded.npy"
SPOTTING_LABEL_SUFFIX = "_spotting_label.npy"
SEGMENTATION_LABEL_SUFFIX = "_segmentation_target.npy"
SEGMENTATION_SCORE_SUFFIX = "_segmentation.npy"
CONFIDENCE_SUFFIX = "_confidence.npy"
DELTA_SUFFIX = "_delta.npy"
SPOTTING_SUFFIXES_AND_NAMES = [
    (DETECTION_SCORE_SUFFIX, COLUMN_DETECTION_SCORE),
    (DETECTION_SCORE_NMS_SUFFIX, COLUMN_DETECTION_SCORE_NMS),
    (DETECTION_SUFFIX, COLUMN_DETECTION),
    (SPOTTING_LABEL_SUFFIX, COLUMN_SPOTTING_LABEL),
    (CONFIDENCE_SUFFIX, COLUMN_CONFIDENCE),
    (DELTA_SUFFIX, COLUMN_DELTA)]
SEGMENTATION_SUFFIXES_AND_NAMES = [
    (SEGMENTATION_SCORE_SUFFIX, COLUMN_SEGMENTATION_SCORE),
    (SEGMENTATION_LABEL_SUFFIX, COLUMN_SEGMENTATION_LABEL)]
MAIN_SUFFIXES = [DETECTION_SCORE_SUFFIX, SEGMENTATION_SCORE_SUFFIX]
SPOTTING_COLUMN_NAMES = [
    COLUMN_DETECTION_SCORE, COLUMN_DETECTION_SCORE_NMS, COLUMN_CONFIDENCE,
    COLUMN_DELTA]
# Etc
SOCCERNET_FEATURES_FREQUENCY = 0.5


class Video:

    def __init__(
            self, relative_path: Path, video_id: str,
            recognized_actions: Optional["VideoSoccerRecognizedActions"],
            spotting_results: Optional[DataFrame],
            segmentation_results: Optional[DataFrame]) -> None:
        self.relative_path = relative_path
        self.video_id = video_id
        self.recognized_actions = recognized_actions
        self.spotting_results = spotting_results
        self.segmentation_results = segmentation_results


class FrameRecognizedActions:

    def __init__(
            self, scores: np.ndarray, detections: np.ndarray,
            labels: np.ndarray) -> None:
        self.scores = scores
        self.detections = detections
        self.labels = labels


VideoSoccerRecognizedActions = Dict[float, FrameRecognizedActions]


def read_videos(results_dir: Path, label_map: LabelMap) -> Dict[Path, Video]:
    video_relative_paths = _get_video_relative_paths(results_dir)
    return {
        video_relative_path: _read_video(
            results_dir, video_relative_path, label_map)
        for video_relative_path in video_relative_paths}


def convert_deltas_to_timestamps(data_frame: DataFrame) -> None:
    """Plotly does not handle timedelta properly when printing, so we convert
    the timedeltas to datetimes, then later specify the tickformat so that
    it prints out properly."""
    data_frame[COLUMN_TIME] = (
            pandas.to_datetime('1970/01/01') +
            pandas.to_timedelta(data_frame[COLUMN_TIME], unit="seconds"))


def _read_video(
        results_dir: Path, video_relative_path: Path,
        label_map: LabelMap) -> Video:
    recognized_actions = _read_video_recognized_actions(
        results_dir, video_relative_path)
    spotting_results = _read_video_spotting_results(
        results_dir, label_map, video_relative_path)
    segmentation_results = _read_video_segmentation_results(
        results_dir, label_map, video_relative_path)
    video_id = _create_video_id(video_relative_path)
    return Video(
        video_relative_path, video_id, recognized_actions, spotting_results,
        segmentation_results)


def _get_video_relative_paths(results_dir: Path) -> List[Path]:
    results_file_paths = results_dir.glob("**/*" + GENERIC_SUFFIX)
    optional_video_relative_paths = [
        _video_relative_path(results_dir, results_file_path)
        for results_file_path in results_file_paths]
    video_relative_paths_set = {
        relative_path for relative_path in optional_video_relative_paths
        if relative_path}
    return list(video_relative_paths_set)


def _video_relative_path(
        results_dir: Path, results_file_path: Path) -> Optional[Path]:
    for main_suffix in MAIN_SUFFIXES:
        if main_suffix in results_file_path.name:
            short_name = results_file_path.name[:(-len(main_suffix))]
            return (results_file_path.parent.relative_to(results_dir) /
                    short_name)
    return None


def _read_video_recognized_actions(
        recognized_actions_dir: Path, video_relative_path: Path) -> \
        Optional[VideoSoccerRecognizedActions]:
    detection_scores_path = _path_with_suffix(
        recognized_actions_dir / video_relative_path, DETECTION_SCORE_SUFFIX)
    labels_path = _path_with_suffix(
        recognized_actions_dir / video_relative_path, SPOTTING_LABEL_SUFFIX)
    if not (detection_scores_path.exists() and labels_path.exists()):
        return None
    detection_scores = np.load(str(detection_scores_path))
    labels = np.load(str(labels_path))
    detection_path = _path_with_suffix(
        recognized_actions_dir / video_relative_path, DETECTION_SUFFIX)
    detection_thresholded = np.load(str(detection_path))
    return {
        _frame_time_from_index(frame_index): FrameRecognizedActions(
            frame_detection_scores, frame_detections, frame_labels)
        for frame_index, (
            frame_detection_scores, frame_detections, frame_labels)
        in enumerate(zip(
            detection_scores, detection_thresholded, labels))}


def _frame_time_from_index(index: int) -> float:
    # This matches the way FeatureExtractorResNet extracts the frames. It
    # starts at frame 0, then grabs a frame approximately every 0.5 seconds.
    return index * SOCCERNET_FEATURES_FREQUENCY


def _read_video_spotting_results(
        results_dir: Path, label_map: LabelMap,
        video_relative_path: Path) -> Optional[DataFrame]:
    base_path = results_dir / video_relative_path
    return _read_partial_results(
        base_path, label_map, SPOTTING_SUFFIXES_AND_NAMES)


def _read_video_segmentation_results(
        results_dir: Path, label_map: LabelMap,
        video_relative_path: Path) -> Optional[DataFrame]:
    base_path = results_dir / video_relative_path
    return _read_partial_results(
        base_path, label_map, SEGMENTATION_SUFFIXES_AND_NAMES)


def _read_partial_results(
        base_path: Path, label_map: LabelMap,
        suffixes_and_names: List[Tuple[str, str]]) -> Optional[DataFrame]:
    paths_and_names = [
        (_path_with_suffix(base_path, suffix), name)
        for suffix, name in suffixes_and_names]
    valid_paths_and_names = [
        (path, name) for path, name in paths_and_names if path.exists()]
    if not valid_paths_and_names:
        return None
    all_data_frames = [
        _read_data_frame(path, label_map, name)
        for path, name in valid_paths_and_names]
    concat_data_frame = pandas.concat(all_data_frames, axis=1, join="inner")
    # This remove duplicate columns that resulted from the above concat,
    # before returning the result.
    return concat_data_frame.loc[:, ~concat_data_frame.columns.duplicated()]


def _read_data_frame(
        path: Path, label_map: LabelMap, column_name: str) -> DataFrame:
    numpy_data = np.load(str(path))
    return _numpy_to_data_frame(numpy_data, label_map, column_name)


def _numpy_to_data_frame(
        values: np.ndarray, label_map: LabelMap,
        values_column_name: str) -> DataFrame:
    n_frames, n_classes = values.shape
    times = [_frame_time_from_index(frame_index)
             for frame_index in range(n_frames)]
    values_in_rows = [
        (times[frame_index], label_map.int_to_label[class_index],
         values[frame_index, class_index])
        for frame_index in range(n_frames)
        for class_index in range(n_classes)]
    return DataFrame(
        values_in_rows, columns=[COLUMN_TIME, COLUMN_CLASS, values_column_name])


def _path_with_suffix(base_path: Path, suffix: str) -> Path:
    new_name = base_path.name + suffix
    return base_path.parent / new_name


def _create_video_id(video_relative_path: Path) -> str:
    return str(video_relative_path).replace("/", "@")
