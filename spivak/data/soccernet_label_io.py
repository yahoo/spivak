# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from tqdm import tqdm

from spivak.data.dataset import VideoDatum, LabelsAndValid
from spivak.data.label_map import LabelMap
from spivak.data.output_names import OUTPUT_DETECTION_SCORE_NMS
from spivak.data.soccernet_constants import EVENT_DICTIONARY_V1, \
    EVENT_DICTIONARY_V2

# SoccerNet dataset types
SOCCERNET_TYPE_ONE = "v1"
SOCCERNET_TYPE_TWO = "v2"
SOCCERNET_TYPE_TWO_CHALLENGE_VALIDATION = "v2_challenge_validation"
SOCCERNET_TYPE_TWO_CHALLENGE = "v2_challenge"
SOCCERNET_TYPE_TWO_CAMERA_SEGMENTATION = "v2_camera_segmentation"
SOCCERNET_TYPE_TWO_SPOTTING_AND_CAMERA_SEGMENTATION = \
    "v2_spotting_and_camera_segmentation"
# Thresholds.
THRESHOLD_ZERO = 0.0
THRESHOLDS = [0.5, 0.7, 0.9]
# File names
RESULTS_JSON = "results_spotting.json"
RESULTS_JSON_THRESHOLDED = "results_spotting_thresholded_%.2f.json"
# JSON keys
KEY_URL_LOCAL = "UrlLocal"
KEY_PREDICTIONS = "predictions"
KEY_ANNOTATIONS = "annotations"
KEY_GAME_TIME = "gameTime"
KEY_LABEL = "label"
KEY_POSITION = "position"
KEY_HALF = "half"
KEY_CONFIDENCE = "confidence"
# These is for reading and writing into JSON files.
HALF_ONE = 1
HALF_TWO = 2
# These are for file naming
PREFIX_HALF_ONE = "1"
PREFIX_HALF_TWO = "2"
SECONDS_PER_MINUTE = 60


class GameSpottingPredictionsWriter:

    def __init__(
            self, label_map: LabelMap, frame_rate: float,
            threshold: float) -> None:
        self._label_map = label_map
        self._frame_rate = frame_rate
        self._threshold = threshold

    def write(
            self, game_path: Path, relative_game_path: Path,
            out_path: Path) -> None:
        half_one_path = game_path / PREFIX_HALF_ONE
        half_two_path = game_path / PREFIX_HALF_TWO
        detections_half_one = np.load(
            f"{half_one_path}_{OUTPUT_DETECTION_SCORE_NMS}.npy")
        detections_half_two = np.load(
            f"{half_two_path}_{OUTPUT_DETECTION_SCORE_NMS}.npy")
        predictions = (
            self._create_half_predictions(detections_half_one, HALF_ONE) +
            self._create_half_predictions(detections_half_two, HALF_TWO)
        )
        output = {
            KEY_URL_LOCAL: str(relative_game_path),
            KEY_PREDICTIONS: predictions}
        with out_path.open("w") as out_file:
            json.dump(output, out_file)

    def _create_half_predictions(
            self, detections: np.ndarray, half: int) -> List[Dict]:
        # detections will be negative in regions with no detections, and can
        # be zero at certain locations (due to how the NMS works), so it's
        # probably good to keep the >= below.
        detection_locations = np.nonzero(detections >= self._threshold)
        # nonzero returns indices as numpy integers, so for simplicity we
        # convert them to regular ints below, as the numpy integers behave
        # differently from regular integers in certain operations.
        return [
            self._create_prediction(
                int(frame_index), int(class_index),
                detections[frame_index, class_index], half)
            for frame_index, class_index in zip(*detection_locations)
        ]

    def _create_prediction(
            self, frame_index: int, class_index: int, confidence: float,
            half: int) -> Dict[str, Any]:
        label = self._label_map.int_to_label[class_index]
        seconds = frame_index / self._frame_rate
        int_milliseconds = round(seconds * 1000.0)
        int_seconds = round(seconds)
        clock_minutes = int_seconds // SECONDS_PER_MINUTE
        clock_seconds = int_seconds - SECONDS_PER_MINUTE * clock_minutes
        game_time = f"{half} - {clock_minutes}:{clock_seconds:02d}"
        return {
            KEY_GAME_TIME: game_time,
            KEY_HALF: str(half),
            KEY_POSITION: str(int_milliseconds),
            KEY_LABEL: label,
            KEY_CONFIDENCE: str(confidence)
        }


class GameSpottingPredictionsReader:

    def __init__(
            self, soccernet_type: str, frame_rate: float,
            num_classes: int) -> None:
        self._frame_rate = frame_rate
        self._num_classes = num_classes
        self._event_dictionary = choose_spotting_event_dictionary(
            soccernet_type)

    def read(
            self, detections_path: Path, len_half_one: int,
            len_half_two: int) -> Tuple[np.ndarray, np.ndarray]:
        return read_game_predictions(
            detections_path, self._event_dictionary, len_half_one,
            len_half_two, self._frame_rate, self._num_classes)


def choose_spotting_event_dictionary(soccernet_type: str) -> Dict[str, int]:
    if soccernet_type == SOCCERNET_TYPE_ONE:
        return EVENT_DICTIONARY_V1
    elif soccernet_type in {
            SOCCERNET_TYPE_TWO, SOCCERNET_TYPE_TWO_CHALLENGE_VALIDATION,
            SOCCERNET_TYPE_TWO_CHALLENGE,
            SOCCERNET_TYPE_TWO_SPOTTING_AND_CAMERA_SEGMENTATION}:
        return EVENT_DICTIONARY_V2
    else:
        raise ValueError(f"Unknown soccernet type: {soccernet_type}")


def write_all_prediction_jsons(
        save_path: Path, video_data: List[VideoDatum],
        label_map: LabelMap, frame_rate: float) -> None:
    game_writer = GameSpottingPredictionsWriter(
        label_map, frame_rate, THRESHOLD_ZERO)
    _write_prediction_jsons(game_writer, save_path, RESULTS_JSON, video_data)
    for threshold in THRESHOLDS:
        results_filename = RESULTS_JSON_THRESHOLDED % threshold
        game_writer_thresholded = GameSpottingPredictionsWriter(
            label_map, frame_rate, threshold)
        _write_prediction_jsons(
            game_writer_thresholded, save_path, results_filename, video_data)


def read_game_predictions(
        json_path: Path, event_dictionary: Dict[str, int],
        len_half_one: int, len_half_two: int, frame_rate: float,
        num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    detections_half_one = np.full((len_half_one, num_classes), - 1.0)
    detections_half_two = np.full((len_half_two, num_classes), - 1.0)
    with json_path.open("r") as json_file:
        detections_dict = json.load(json_file)
    for annotation in detections_dict[KEY_PREDICTIONS]:
        event = annotation[KEY_LABEL]
        if event not in event_dictionary:
            continue
        label_index = event_dictionary[event]
        confidence = float(annotation[KEY_CONFIDENCE])
        # Time resolution is only up to 1 second in the game time field,
        # so we get the time from position instead, which has finer
        # resolution, up to the millisecond in theory.
        time_str = annotation[KEY_GAME_TIME]
        half = _half_from_time_str(time_str)
        position = int(annotation[KEY_POSITION])
        frame_index = _frame_index_from_position(position, frame_rate)
        if half == HALF_ONE:
            frame_index = min(frame_index, len_half_one - 1)
            detections_half_one[frame_index, label_index] = confidence
        elif half == HALF_TWO:
            frame_index = min(frame_index, len_half_two - 1)
            detections_half_two[frame_index, label_index] = confidence
    return detections_half_one, detections_half_two


def read_game_labels(
        json_path: Optional[Path], event_dictionary: Dict[str, int],
        len_half_one: int, len_half_two: int, frame_rate: float,
        num_classes: int) -> Tuple[LabelsAndValid, LabelsAndValid]:
    """
    Transforms the labels from the json format to a numpy array with size
    (num_frames, num_classes). Every element is set to zero except those
    that correspond to events (those that are at the frame and class of a
    given event), which are set to 1.
    """
    labels_half_one = np.zeros((len_half_one, num_classes))
    labels_half_two = np.zeros((len_half_two, num_classes))
    if not (json_path and json_path.exists()):
        return (labels_half_one, False), (labels_half_two, False)
    with json_path.open("r") as json_file:
        labels_dict = json.load(json_file)
    for annotation in labels_dict[KEY_ANNOTATIONS]:
        event = annotation[KEY_LABEL]
        if event not in event_dictionary:
            continue
        label_index = event_dictionary[event]
        time_str = annotation[KEY_GAME_TIME]
        half = _half_from_time_str(time_str)
        frame_index = _frame_index_from_time_str(time_str, frame_rate)
        if half == HALF_ONE:
            frame_index = min(frame_index, len_half_one - 1)
            labels_half_one[frame_index, label_index] = 1.0
        elif half == HALF_TWO:
            frame_index = min(frame_index, len_half_two - 1)
            labels_half_two[frame_index, label_index] = 1.0
    return (labels_half_one, True), (labels_half_two, True)


def read_game_change_labels(
        json_path: Optional[Path], event_dictionary: Dict[str, int],
        len_half_one: int, len_half_two: int, frame_rate: float,
        num_classes: int) -> Tuple[LabelsAndValid, LabelsAndValid]:
    labels_half_one = np.zeros((len_half_one, num_classes))
    labels_half_two = np.zeros((len_half_two, num_classes))
    if not (json_path and json_path.exists()):
        return (labels_half_one, False), (labels_half_two, False)
    with json_path.open("r") as json_file:
        labels_dict = json.load(json_file)
    for annotation in labels_dict[KEY_ANNOTATIONS]:
        _read_game_change_annotation(
            annotation, event_dictionary, frame_rate, labels_half_one,
            labels_half_two, len_half_one, len_half_two)
    return (labels_half_one, True), (labels_half_two, True)


def segmentation_targets_from_change_labels(changes: np.ndarray) -> np.ndarray:
    """This code should be similar to the original SoccerNet code for
    converting from change labels to segmentation targets."""
    num_frames = changes.shape[0]
    flipped_changes = np.flip(changes, 0)
    flipped_labels = flipped_changes
    active_label = 0
    for frame_index in range(num_frames):
        frame_flipped_changes = flipped_changes[frame_index, :]
        change_locations = np.where(frame_flipped_changes == 1.0)[0]
        if frame_flipped_changes[change_locations].size > 0:
            active_label = change_locations[0]
        flipped_labels[frame_index, active_label] = 1.0
    return np.flip(flipped_labels, 0)


def _read_game_change_annotation(
        annotation: Dict, event_dictionary: Dict[str, int], frame_rate: float,
        labels_half_one: np.ndarray, labels_half_two: np.ndarray,
        len_half_one: int, len_half_two: int) -> None:
    event = annotation[KEY_LABEL]
    # Blank camera type labels are often present. SoccerNet code ignores
    # them, so we do the same here.
    if not event:
        return
    label_index = event_dictionary[event]
    time_str = annotation[KEY_GAME_TIME]
    half = _half_from_time_str(time_str)
    frame_index = _frame_index_from_time_str(time_str, frame_rate)
    if half == HALF_ONE:
        frame_index = min(frame_index, len_half_one - 1)
        # If there is already a label set at this frame, keep the
        # existing one, as it should indicate what camera type came
        # before it.
        if not np.any(labels_half_one[frame_index]):
            labels_half_one[frame_index, label_index] = 1.0
    elif half == HALF_TWO:
        frame_index = min(frame_index, len_half_two - 1)
        if not np.any(labels_half_two[frame_index]):
            labels_half_two[frame_index, label_index] = 1.0


def _half_from_time_str(time_str: str) -> int:
    return int(time_str[0])


def _frame_index_from_time_str(time_str: str, frame_rate: float) -> int:
    # Get whatever comes after the dash.
    minutes_and_seconds_str = time_str.partition("-")[2].strip()
    # Split into two strings, then convert to int.
    minutes_and_seconds = minutes_and_seconds_str.partition(":")
    minutes = int(minutes_and_seconds[0])
    seconds = int(minutes_and_seconds[2])
    return round(frame_rate * (seconds + 60 * minutes))


def _frame_index_from_position(position: int, frame_rate: float) -> int:
    return round(frame_rate * position / 1000.0)


def _write_prediction_jsons(
        game_writer: GameSpottingPredictionsWriter, save_path: Path,
        results_filename: str, video_data: List[VideoDatum]) -> None:
    # Get the directory containing each game
    relative_game_paths = {
        video_datum.relative_path.parent for video_datum in video_data}
    for relative_game_path in tqdm(relative_game_paths):
        game_path = save_path / relative_game_path
        out_path = game_path / results_filename
        game_writer.write(game_path, relative_game_path, out_path)
