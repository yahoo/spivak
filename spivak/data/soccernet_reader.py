# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import csv
import json
import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from spivak.data.dataset import Dataset, DefaultVideoDatum, read_num_frames, \
    LabelsFromTaskDict, Task, LabelsAndValid
from spivak.data.dataset_splits import SPLIT_KEY_TRAIN, SPLIT_KEY_VALIDATION, \
    SPLIT_KEY_TEST, SPLIT_KEY_UNLABELED
from spivak.data.soccernet_constants import LABEL_FILE_NAME, \
    LABEL_FILE_NAME_V2, LABEL_FILE_NAME_V2_CAMERAS, CAMERA_DICTIONARY
from spivak.data.soccernet_label_io import read_game_labels, PREFIX_HALF_ONE, \
    PREFIX_HALF_TWO, read_game_change_labels, \
    choose_spotting_event_dictionary, SOCCERNET_TYPE_ONE, SOCCERNET_TYPE_TWO, \
    SOCCERNET_TYPE_TWO_CHALLENGE_VALIDATION, SOCCERNET_TYPE_TWO_CHALLENGE, \
    SOCCERNET_TYPE_TWO_CAMERA_SEGMENTATION, \
    SOCCERNET_TYPE_TWO_SPOTTING_AND_CAMERA_SEGMENTATION

# For reading V2 game lists.
V2_GAMES_JSON_TRAIN = "SoccerNetGamesTrain.json"
V2_GAMES_JSON_VALIDATION = "SoccerNetGamesValid.json"
V2_GAMES_JSON_TEST = "SoccerNetGamesTest.json"
V2_GAMES_JSON_CHALLENGE = "SoccerNetGamesChallenge.json"
V2_GAMES_JSONS = {
    SPLIT_KEY_TRAIN: [V2_GAMES_JSON_TRAIN],
    SPLIT_KEY_VALIDATION: [V2_GAMES_JSON_VALIDATION],
    SPLIT_KEY_TEST: [V2_GAMES_JSON_TEST],
    SPLIT_KEY_UNLABELED: [V2_GAMES_JSON_CHALLENGE],
}
V2_CHALLENGE_VALIDATION_GAMES_JSONS = {
    SPLIT_KEY_TRAIN: [V2_GAMES_JSON_TRAIN, V2_GAMES_JSON_TEST],
    # We run validation with V2_GAMES_JSON_VALIDATION instead of
    # V2_GAMES_JSON_TEST so that we can compare validation results across
    # different setups. Namely, we can compare the validation metrics from
    # this setup with the standard one from V2_GAMES_JSONS above.
    SPLIT_KEY_VALIDATION: [V2_GAMES_JSON_VALIDATION],
    SPLIT_KEY_UNLABELED: [V2_GAMES_JSON_CHALLENGE]
}
V2_CHALLENGE_GAMES_JSONS = {
    SPLIT_KEY_TRAIN: [
        V2_GAMES_JSON_TRAIN, V2_GAMES_JSON_VALIDATION, V2_GAMES_JSON_TEST],
    SPLIT_KEY_UNLABELED: [V2_GAMES_JSON_CHALLENGE]
}
V2_CAMERA_SEGMENTATION_GAMES_JSONS = {
    SPLIT_KEY_TRAIN: ["SoccerNetCameraChangesTrain.json"],
    SPLIT_KEY_VALIDATION: ["SoccerNetCameraChangesValid.json"],
    SPLIT_KEY_TEST: ["SoccerNetCameraChangesTest.json"],
    SPLIT_KEY_UNLABELED: ["SoccerNetCameraChangesChallenge.json"],
}
V2_SPOTTING_AND_CAMERA_SEGMENTATION_GAMES_CSVS = [
    "SoccerNetSpottingAndCameraChangesLarge.csv"]
GAMES_CSV_COLUMN_GAME = "DIRECTORY"
GAMES_CSV_COLUMN_SPLIT_KEY = "SPLIT"
V1_SPLIT_KEYS = {SPLIT_KEY_TRAIN, SPLIT_KEY_VALIDATION, SPLIT_KEY_TEST}
V2_SPOTTING_AND_CAMERA_SEGMENTATION_SPLIT_KEYS = {
    SPLIT_KEY_TRAIN, SPLIT_KEY_VALIDATION, SPLIT_KEY_TEST}


class SoccerNetReader:

    def __init__(
            self, soccernet_video_data_reader: "SoccerNetVideoDataReader",
            num_chunk_frames: int) -> None:
        self._soccernet_video_data_reader = soccernet_video_data_reader
        self._num_chunk_frames = num_chunk_frames

    def read(self, split_key: str) -> Optional[Dataset]:
        if not self._soccernet_video_data_reader.has_split(split_key):
            return None
        video_data = self._soccernet_video_data_reader.read(split_key)
        num_features = video_data[0].num_features
        num_classes_from_task = video_data[0].num_classes_from_task
        input_shape = (self._num_chunk_frames, num_features, 1)
        return Dataset(video_data, input_shape, num_classes_from_task)


class SoccerNetVideoDataReader:

    def __init__(
            self, game_label_reader: "GameLabelsFromTaskDictReader",
            game_paths_reader: "GamePathsReader") -> None:
        self._game_label_reader = game_label_reader
        self._game_paths_reader = game_paths_reader

    def has_split(self, split_key: str) -> bool:
        return self._game_paths_reader.has_split(split_key)

    def read(self, split_key: str) -> List[DefaultVideoDatum]:
        valid_game_paths = self._game_paths_reader.read_valid(split_key)
        # Load the dataset from the original SoccerNet files.
        video_data = []
        for game_paths in valid_game_paths:
            # Read the number of frames for each video.
            num_video_frames_one = read_num_frames(
                game_paths.features_one)
            num_video_frames_two = read_num_frames(
                game_paths.features_two)
            # Read all the labels for the two videos.
            labels_from_task_dict_one, labels_from_task_dict_two = \
                self._game_label_reader.read(
                    game_paths, num_video_frames_one, num_video_frames_two)
            # Build VideoDatum for each of the two halves.
            video_datum_one = DefaultVideoDatum(
                game_paths.features_one,
                game_paths.relative / PREFIX_HALF_ONE,
                labels_from_task_dict_one, num_video_frames_one)
            video_data.append(video_datum_one)
            video_datum_two = DefaultVideoDatum(
                game_paths.features_two,
                game_paths.relative / PREFIX_HALF_TWO,
                labels_from_task_dict_two, num_video_frames_two)
            video_data.append(video_datum_two)
        if not video_data:
            raise ValueError("Did not find any features for the dataset")
        return video_data


class GameLabelsFromTaskDictReader:

    def __init__(
            self, game_one_hot_label_readers: Dict[
                Task, "GameOneHotLabelReaderInterface"]) -> None:
        self._game_one_hot_label_readers = game_one_hot_label_readers

    def read(
            self, game_paths: "GamePaths", len_half_one: int, len_half_two: int
    ) -> Tuple[LabelsFromTaskDict, LabelsFromTaskDict]:
        labels_from_task_dict_one = {}
        labels_from_task_dict_two = {}
        for task in self._game_one_hot_label_readers:
            task_one_hot_label_reader = self._game_one_hot_label_readers[task]
            labels_one, labels_two = task_one_hot_label_reader.read(
                game_paths.labels.get(task), len_half_one, len_half_two)
            labels_from_task_dict_one[task] = labels_one
            labels_from_task_dict_two[task] = labels_two
        return labels_from_task_dict_one, labels_from_task_dict_two


class GameOneHotLabelReaderInterface(metaclass=ABCMeta):

    @abstractmethod
    def read(self, game_labels_path: Optional[Path], len_half_one: int,
             len_half_two: int) -> Tuple[LabelsAndValid, LabelsAndValid]:
        pass


class GameOneHotSpottingLabelReader(GameOneHotLabelReaderInterface):

    def __init__(
            self, soccernet_type: str, frame_rate: float,
            num_classes: int) -> None:
        self._frame_rate = frame_rate
        self._num_classes = num_classes
        self._event_dictionary = choose_spotting_event_dictionary(
            soccernet_type)

    def read(self, game_labels_path: Optional[Path], len_half_one: int,
             len_half_two: int) -> Tuple[LabelsAndValid, LabelsAndValid]:
        return read_game_labels(
            game_labels_path, self._event_dictionary, len_half_one,
            len_half_two, self._frame_rate, self._num_classes)


class GameOneHotCameraChangeLabelReader(GameOneHotLabelReaderInterface):

    def __init__(self, frame_rate: float, num_classes: int) -> None:
        self._frame_rate = frame_rate
        self._num_classes = num_classes

    def read(self, game_labels_path: Optional[Path], len_half_one: int,
             len_half_two: int) -> Tuple[LabelsAndValid, LabelsAndValid]:
        return read_game_change_labels(
            game_labels_path, CAMERA_DICTIONARY, len_half_one, len_half_two,
            self._frame_rate, self._num_classes)


class GamePaths:

    def __init__(
            self, features_one: Path, features_two: Path,
            labels: Dict[Task, Path], relative: Path) -> None:
        self.features_one = features_one
        self.features_two = features_two
        self.labels = labels
        self.relative = relative


class GamePathsReader:

    def __init__(
            self, soccernet_type: str, feature_name: str, features_dir: Path,
            labels_dir: Path, splits_dir: Path) -> None:
        self._soccernet_type = soccernet_type
        self._features_dir = features_dir
        self._labels_dir = labels_dir
        self._splits_dir = splits_dir
        self._feature_name = feature_name
        self._label_file_names = \
            GamePathsReader._choose_label_file_names(soccernet_type)

    def has_split(self, split_key: str) -> bool:
        if self._soccernet_type == SOCCERNET_TYPE_ONE:
            return split_key in V1_SPLIT_KEYS
        elif self._soccernet_type == SOCCERNET_TYPE_TWO:
            return split_key in V2_GAMES_JSONS
        elif self._soccernet_type == SOCCERNET_TYPE_TWO_CHALLENGE_VALIDATION:
            return split_key in V2_CHALLENGE_VALIDATION_GAMES_JSONS
        elif self._soccernet_type == SOCCERNET_TYPE_TWO_CHALLENGE:
            return split_key in V2_CHALLENGE_GAMES_JSONS
        elif self._soccernet_type == SOCCERNET_TYPE_TWO_CAMERA_SEGMENTATION:
            return split_key in V2_CAMERA_SEGMENTATION_GAMES_JSONS
        elif self._soccernet_type == \
                SOCCERNET_TYPE_TWO_SPOTTING_AND_CAMERA_SEGMENTATION:
            return split_key in V2_SPOTTING_AND_CAMERA_SEGMENTATION_SPLIT_KEYS
        else:
            raise ValueError(
                f"Unrecognized soccernet type: {self._soccernet_type}")

    def read_valid(self, split_key: str) -> List[GamePaths]:
        all_game_paths = self.read(split_key)
        valid_game_paths = []
        for game_paths in all_game_paths:
            if (game_paths.features_one.exists() and
                    game_paths.features_two.exists()):
                valid_game_paths.append(game_paths)
            else:
                logging.warning(
                    f"Missing at least one half of the game: "
                    f"{game_paths.features_one} and/or "
                    f"{game_paths.features_two}")
        return valid_game_paths

    def read(self, split_key: str) -> List[GamePaths]:
        if self._soccernet_type in {
                SOCCERNET_TYPE_ONE, SOCCERNET_TYPE_TWO,
                SOCCERNET_TYPE_TWO_CHALLENGE_VALIDATION,
                SOCCERNET_TYPE_TWO_CHALLENGE,
                SOCCERNET_TYPE_TWO_CAMERA_SEGMENTATION}:
            game_list = GamePathsReader._read_standard_game_list(
                self._soccernet_type, self._splits_dir, split_key)
            all_game_paths = [
                self._create_standard_game_paths(game) for game in game_list]
        elif self._soccernet_type == \
                SOCCERNET_TYPE_TWO_SPOTTING_AND_CAMERA_SEGMENTATION:
            all_game_paths = \
                self._read_game_paths_v2_spotting_and_segmentation(split_key)
        else:
            raise ValueError(
                f"Unrecognized soccernet type: {self._soccernet_type}")
        return all_game_paths

    @staticmethod
    def read_game_list_v2(splits_dir: Path, split_key: str) -> List[Path]:
        json_file_names = V2_GAMES_JSONS[split_key]
        json_paths = [
            splits_dir / json_file_name for json_file_name in json_file_names]
        return GamePathsReader._read_game_list_from_jsons(json_paths)

    @staticmethod
    def read_game_list_v2_camera_segmentation(
            splits_dir: Path, split_key: str) -> List[Path]:
        json_file_names = V2_CAMERA_SEGMENTATION_GAMES_JSONS[split_key]
        json_paths = [
            splits_dir / json_file_name for json_file_name in json_file_names]
        return GamePathsReader._read_game_list_from_jsons(json_paths)

    @staticmethod
    def read_game_list_v2_challenge_validation(
            splits_dir: Path, split_key: str) -> List[Path]:
        json_file_names = V2_CHALLENGE_VALIDATION_GAMES_JSONS[split_key]
        json_paths = [
            splits_dir / json_file_name for json_file_name in json_file_names]
        return GamePathsReader._read_game_list_from_jsons(json_paths)

    @staticmethod
    def read_game_list_v2_challenge(
            splits_dir: Path, split_key: str) -> List[Path]:
        json_file_names = V2_CHALLENGE_GAMES_JSONS[split_key]
        json_paths = [
            splits_dir / json_file_name for json_file_name in json_file_names]
        return GamePathsReader._read_game_list_from_jsons(json_paths)

    @staticmethod
    def _read_standard_game_list(
            soccernet_type: str, splits_dir: Path,
            split_key: str) -> List[Path]:
        if soccernet_type in {SOCCERNET_TYPE_ONE, SOCCERNET_TYPE_TWO}:
            return GamePathsReader.read_game_list_v2(splits_dir, split_key)
        elif soccernet_type == SOCCERNET_TYPE_TWO_CHALLENGE_VALIDATION:
            return GamePathsReader.read_game_list_v2_challenge_validation(
                splits_dir, split_key)
        elif soccernet_type == SOCCERNET_TYPE_TWO_CHALLENGE:
            return GamePathsReader.read_game_list_v2_challenge(
                splits_dir, split_key)
        elif soccernet_type == SOCCERNET_TYPE_TWO_CAMERA_SEGMENTATION:
            return GamePathsReader.read_game_list_v2_camera_segmentation(
                splits_dir, split_key)
        else:
            raise ValueError(
                f"Unrecognized standard soccernet type: {soccernet_type}")

    def _create_standard_game_paths(self, game: Path):
        game_features_dir = self._features_dir / game
        features_one = game_features_dir / (
            f"{PREFIX_HALF_ONE}_{self._feature_name}.npy")
        features_two = game_features_dir / (
            f"{PREFIX_HALF_TWO}_{self._feature_name}.npy")
        base_labels = self._labels_dir / game
        labels = {
            task: base_labels / self._label_file_names[task]
            for task in self._label_file_names}
        return GamePaths(features_one, features_two, labels, game)

    @staticmethod
    def _read_game_list_from_jsons(json_paths: List[Path]) -> List[Path]:
        game_list = []
        for json_path in json_paths:
            json_game_list = GamePathsReader._read_game_list_from_single_json(
                json_path)
            game_list.extend(json_game_list)
        return game_list

    @staticmethod
    def _read_game_list_from_single_json(json_path: Path) -> List[Path]:
        game_list = []
        with json_path.open("r") as json_file:
            dictionary = json.load(json_file)
        for championship in dictionary:
            for season in dictionary[championship]:
                for game in dictionary[championship][season]:
                    game_list.append(Path(championship, season, game))
        return game_list

    def _read_game_paths_v2_spotting_and_segmentation(
            self, split_key: str) -> List[GamePaths]:
        csv_file_names = V2_SPOTTING_AND_CAMERA_SEGMENTATION_GAMES_CSVS
        csv_paths = [
            self._splits_dir / csv_file_name
            for csv_file_name in csv_file_names]
        all_splits_game_paths = self._read_game_paths_from_csvs(csv_paths)
        return all_splits_game_paths[split_key]

    def _read_game_paths_from_csvs(
            self, csv_paths: List[Path]) -> Dict[str, List[GamePaths]]:
        all_splits_game_paths = defaultdict(list)
        for csv_path in csv_paths:
            csv_all_splits_game_paths = self._read_game_paths_from_single_csv(
                csv_path)
            for split_key in csv_all_splits_game_paths:
                all_splits_game_paths[split_key].extend(
                    csv_all_splits_game_paths[split_key])
        return all_splits_game_paths

    def _read_game_paths_from_single_csv(
            self, csv_path: Path) -> Dict[str, List[GamePaths]]:
        all_splits_game_paths = defaultdict(list)
        with csv_path.open("r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                game = Path(row[GAMES_CSV_COLUMN_GAME])
                game_features_dir = self._features_dir / game
                features_one = game_features_dir / (
                    f"{PREFIX_HALF_ONE}_{self._feature_name}.npy")
                features_two = game_features_dir / (
                    f"{PREFIX_HALF_TWO}_{self._feature_name}.npy")
                base_labels = self._labels_dir / game
                labels = {
                    task: base_labels / self._label_file_names[task]
                    for task in self._label_file_names
                    if int(row[task.name])}
                game_paths = GamePaths(features_one, features_two, labels, game)
                split_key = row[GAMES_CSV_COLUMN_SPLIT_KEY]
                all_splits_game_paths[split_key].append(game_paths)
        return all_splits_game_paths

    @staticmethod
    def _choose_label_file_names(soccernet_type: str) -> Dict[Task, str]:
        if soccernet_type == SOCCERNET_TYPE_ONE:
            return {Task.SPOTTING: LABEL_FILE_NAME}
        elif soccernet_type in {
                SOCCERNET_TYPE_TWO, SOCCERNET_TYPE_TWO_CHALLENGE_VALIDATION,
                SOCCERNET_TYPE_TWO_CHALLENGE}:
            return {Task.SPOTTING: LABEL_FILE_NAME_V2}
        elif soccernet_type == SOCCERNET_TYPE_TWO_CAMERA_SEGMENTATION:
            return {Task.SEGMENTATION: LABEL_FILE_NAME_V2_CAMERAS}
        elif soccernet_type == \
                SOCCERNET_TYPE_TWO_SPOTTING_AND_CAMERA_SEGMENTATION:
            return {
                Task.SPOTTING: LABEL_FILE_NAME_V2,
                Task.SEGMENTATION: LABEL_FILE_NAME_V2_CAMERAS}
        else:
            raise ValueError(
                f"Unrecognized soccernet type: {soccernet_type}")
