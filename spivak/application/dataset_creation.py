# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import math
from pathlib import Path
from typing import List, Dict, Optional

from spivak.application.argument_parser import SharedArgs, dir_str_to_path, \
    DATASET_TYPE_SOCCERNET, DATASET_TYPE_CUSTOM_SPOTTING, \
    DATASET_TYPE_SOCCERNET_V2, DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION, \
    DATASET_TYPE_SOCCERNET_V2_CHALLENGE, \
    DATASET_TYPE_SOCCERNET_V2_CAMERA_SEGMENTATION, \
    DATASET_TYPE_CUSTOM_SEGMENTATION, \
    DATASET_TYPE_CUSTOM_SPOTTING_AND_SEGMENTATION, \
    DATASET_TYPE_SOCCERNET_V2_SPOTTING_AND_CAMERA_SEGMENTATION
from spivak.data.custom_dataset_reader import CustomDatasetReader, \
    VideoLabelReader, GenericLabelFileReader, read_label_groups
from spivak.data.dataset import Dataset, Task
from spivak.data.dataset_splits import load_or_create_splits, Splits, \
    create_features_paths, SplitPathsProvider
from spivak.data.label_map import LabelMap
from spivak.data.soccernet_reader import SOCCERNET_TYPE_ONE, \
    SOCCERNET_TYPE_TWO, SOCCERNET_TYPE_TWO_CHALLENGE_VALIDATION, \
    SOCCERNET_TYPE_TWO_CHALLENGE, SOCCERNET_TYPE_TWO_CAMERA_SEGMENTATION, \
    SOCCERNET_TYPE_TWO_SPOTTING_AND_CAMERA_SEGMENTATION, \
    GameOneHotSpottingLabelReader, GameLabelsFromTaskDictReader, \
    GamePathsReader, SoccerNetReader, GameOneHotCameraChangeLabelReader, \
    GameOneHotLabelReaderInterface, SoccerNetVideoDataReader

LABEL_MAP_CSV_SPOTTING = "spotting_labels.csv"
LABEL_MAP_CSV_SEGMENTATION = "segmentation_labels.csv"
SPLITS_CSV = "video_splits.csv"
LABEL_GROUP_JSON_FILES = {
    Task.SPOTTING: "spotting_label_groups.json",
    Task.SEGMENTATION: "segmentation_label_groups.json"
}


def create_label_maps(args: SharedArgs) -> Dict[Task, LabelMap]:
    config_dir = dir_str_to_path(args.config_dir)
    label_maps = {}
    optional_spotting_label_map = read_spotting_label_map(config_dir)
    if optional_spotting_label_map:
        label_maps[Task.SPOTTING] = optional_spotting_label_map
    optional_segmentation_label_map = read_segmentation_label_map(config_dir)
    if optional_segmentation_label_map:
        label_maps[Task.SEGMENTATION] = optional_segmentation_label_map
    return label_maps


def read_spotting_label_map(config_dir: Path) -> Optional[LabelMap]:
    spotting_path = config_dir / LABEL_MAP_CSV_SPOTTING
    if not spotting_path.exists():
        return None
    return LabelMap.read_label_map(spotting_path)


def read_segmentation_label_map(config_dir: Path) -> Optional[LabelMap]:
    segmentation_path = config_dir / LABEL_MAP_CSV_SEGMENTATION
    if not segmentation_path.exists():
        return None
    return LabelMap.read_label_map(segmentation_path)


def create_dataset(
        args: SharedArgs, split_key: str,
        label_maps: Dict[Task, LabelMap]) -> Dataset:
    datasets = create_datasets(args, [split_key], label_maps)
    assert len(datasets) == 1
    return datasets[0]


def create_datasets(
        args: SharedArgs, split_keys: List[str],
        label_maps: Dict[Task, LabelMap]) -> List[Optional[Dataset]]:
    if args.dataset_type in {
            DATASET_TYPE_SOCCERNET, DATASET_TYPE_SOCCERNET_V2,
            DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION,
            DATASET_TYPE_SOCCERNET_V2_CHALLENGE,
            DATASET_TYPE_SOCCERNET_V2_CAMERA_SEGMENTATION,
            DATASET_TYPE_SOCCERNET_V2_SPOTTING_AND_CAMERA_SEGMENTATION}:
        return _create_soccernet_datasets(args, split_keys, label_maps)
    elif args.dataset_type in {
            DATASET_TYPE_CUSTOM_SPOTTING, DATASET_TYPE_CUSTOM_SEGMENTATION,
            DATASET_TYPE_CUSTOM_SPOTTING_AND_SEGMENTATION}:
        return _create_custom_datasets(args, split_keys, label_maps)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")


def _create_soccernet_datasets(
        args: SharedArgs, split_keys: List[str],
        label_maps: Dict[Task, LabelMap]) -> List[Optional[Dataset]]:
    soccernet_reader = _create_soccernet_reader(args, label_maps)
    return [soccernet_reader.read(split_key) for split_key in split_keys]


def _create_soccernet_reader(
        args: SharedArgs, label_maps: Dict[Task, LabelMap]
) -> SoccerNetReader:
    soccernet_video_data_reader = create_soccernet_video_data_reader(
        args, label_maps)
    chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    return SoccerNetReader(soccernet_video_data_reader, chunk_frames)


def create_soccernet_video_data_reader(
        args: SharedArgs, label_maps: Dict[Task, LabelMap]
) -> SoccerNetVideoDataReader:
    if not args.splits_dir or not args.labels_dir or not args.features_dir:
        raise ValueError(f"Must have splits_dir, labels_dir and features_dir")
    features_dir = dir_str_to_path(args.features_dir)
    soccernet_type = _get_soccernet_type(args.dataset_type)
    game_one_hot_label_readers = _create_game_one_hot_label_readers(
        soccernet_type, args.frame_rate, label_maps)
    game_label_reader = GameLabelsFromTaskDictReader(game_one_hot_label_readers)
    splits_dir = dir_str_to_path(args.splits_dir)
    labels_dir = dir_str_to_path(args.labels_dir)
    game_paths_reader = GamePathsReader(
        soccernet_type, args.feature_name, features_dir, labels_dir, splits_dir)
    return SoccerNetVideoDataReader(game_label_reader, game_paths_reader)


def _create_game_one_hot_label_readers(
        soccernet_type: str, frame_rate: float,
        label_maps: Dict[Task, LabelMap]
) -> Dict[Task, GameOneHotLabelReaderInterface]:
    game_one_hot_label_readers = {}
    if soccernet_type in {
            SOCCERNET_TYPE_TWO_CAMERA_SEGMENTATION,
            SOCCERNET_TYPE_TWO_SPOTTING_AND_CAMERA_SEGMENTATION}:
        game_one_hot_label_readers[Task.SEGMENTATION] = \
            GameOneHotCameraChangeLabelReader(
                frame_rate, label_maps[Task.SEGMENTATION].num_classes())
    if soccernet_type in {
            SOCCERNET_TYPE_ONE, SOCCERNET_TYPE_TWO,
            SOCCERNET_TYPE_TWO_CHALLENGE_VALIDATION,
            SOCCERNET_TYPE_TWO_CHALLENGE,
            SOCCERNET_TYPE_TWO_SPOTTING_AND_CAMERA_SEGMENTATION}:
        game_one_hot_label_readers[Task.SPOTTING] = \
            GameOneHotSpottingLabelReader(
                soccernet_type, frame_rate,
                label_maps[Task.SPOTTING].num_classes())
    return game_one_hot_label_readers


def _get_soccernet_type(dataset_type: str) -> str:
    if dataset_type == DATASET_TYPE_SOCCERNET:
        return SOCCERNET_TYPE_ONE
    elif dataset_type == DATASET_TYPE_SOCCERNET_V2:
        return SOCCERNET_TYPE_TWO
    elif dataset_type == DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION:
        return SOCCERNET_TYPE_TWO_CHALLENGE_VALIDATION
    elif dataset_type == DATASET_TYPE_SOCCERNET_V2_CHALLENGE:
        return SOCCERNET_TYPE_TWO_CHALLENGE
    elif dataset_type == DATASET_TYPE_SOCCERNET_V2_CAMERA_SEGMENTATION:
        return SOCCERNET_TYPE_TWO_CAMERA_SEGMENTATION
    elif (dataset_type ==
          DATASET_TYPE_SOCCERNET_V2_SPOTTING_AND_CAMERA_SEGMENTATION):
        return SOCCERNET_TYPE_TWO_SPOTTING_AND_CAMERA_SEGMENTATION
    else:
        raise ValueError(f"Not a SoccerNet dataset type: {dataset_type}")


def _create_custom_datasets(
        args: SharedArgs, split_keys: List[str],
        label_maps: Dict[Task, LabelMap]) -> List[Dataset]:
    tasks = _tasks_from_custom_dataset_type(args.dataset_type)
    dataset_reader = _create_custom_dataset_reader(args, label_maps, tasks)
    split_paths_provider = _create_split_paths_provider(args, tasks)
    return [_create_custom_dataset(
        dataset_reader, split_paths_provider, split_key)
        for split_key in split_keys]


def _create_custom_dataset_reader(
        args: SharedArgs, label_maps: Dict[Task, LabelMap],
        tasks: List[Task]) -> CustomDatasetReader:
    config_dir = dir_str_to_path(args.config_dir)
    video_label_readers = {}
    for task in tasks:
        label_groups_path = config_dir / LABEL_GROUP_JSON_FILES[task]
        optional_label_groups = read_label_groups(label_groups_path)
        label_file_reader = GenericLabelFileReader(
            label_maps[task], optional_label_groups, args.frame_rate)
        video_label_readers[task] = VideoLabelReader(label_file_reader)
    chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    return CustomDatasetReader(video_label_readers, chunk_frames)


def _tasks_from_custom_dataset_type(dataset_type: str) -> List[Task]:
    if dataset_type == DATASET_TYPE_CUSTOM_SPOTTING:
        tasks = [Task.SPOTTING]
    elif dataset_type == DATASET_TYPE_CUSTOM_SEGMENTATION:
        tasks = [Task.SEGMENTATION]
    elif dataset_type == DATASET_TYPE_CUSTOM_SPOTTING_AND_SEGMENTATION:
        tasks = [Task.SPOTTING, Task.SEGMENTATION]
    else:
        raise ValueError(
            f"Unrecognized dataset type in _create_custom_dataset_reader():"
            f" {dataset_type}")
    return tasks


def _create_split_paths_provider(
        args: SharedArgs, tasks: List[Task]) -> SplitPathsProvider:
    features_dir = dir_str_to_path(args.features_dir)
    features_paths = create_features_paths(features_dir, args.feature_name)
    labels_dir_dict = dict()
    if args.labels_dir:
        labels_dir = dir_str_to_path(args.labels_dir)
        # TODO: allow multiple labels_dir so that labels for each task can be
        #  read. For now, we assume there is only one task.
        assert len(tasks) == 1
        labels_dir_dict[tasks[0]] = labels_dir
    splits = _create_splits(args, features_paths, labels_dir_dict)
    return SplitPathsProvider(features_paths, labels_dir_dict, splits)


def _create_splits(
        args: SharedArgs, features_paths: List[Path],
        labels_dir_dict: Dict[Task, Path]) -> Splits:
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(exist_ok=True)
    splits_path = splits_dir / SPLITS_CSV
    return load_or_create_splits(features_paths, labels_dir_dict, splits_path)


def _create_custom_dataset(
        dataset_reader: CustomDatasetReader,
        split_paths_provider: SplitPathsProvider, split_key: str) -> Dataset:
    split_features_paths, split_labels_path_dicts = \
        split_paths_provider.provide(split_key)
    return dataset_reader.read(split_features_paths, split_labels_path_dicts)
