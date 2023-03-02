# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

# For reading/writing the splits
from spivak.data.dataset import Task

FIELD_VIDEO_NAME = "video_name"
FIELD_SPLIT_KEY = "split_key"
FIELD_NAMES = [FIELD_VIDEO_NAME, FIELD_SPLIT_KEY]
# Fractions used for splitting the labeled data into train, validation,
# and test sets.
FRACTION_TRAIN = 0.8
FRACTION_VALIDATION = 0.1
# Split types
SPLIT_KEY_TEST = "test"
SPLIT_KEY_TRAIN = "train"
SPLIT_KEY_VALIDATION = "validation"
# For videos that have no ground truth. Used just for visualizing results.
SPLIT_KEY_UNLABELED = "unlabeled"
ALL_SPLIT_KEYS = [
    SPLIT_KEY_TRAIN, SPLIT_KEY_VALIDATION, SPLIT_KEY_TEST, SPLIT_KEY_UNLABELED]
# FRACTION_TEST is implicitly equal to 1 - FRACTION_TRAIN - FRACTION_VALIDATION
JSON_EXTENSION = ".json"

Split = List[str]
Splits = Dict[str, Split]


class SplitPathsProvider:

    def __init__(
            self, features_paths: List[Path], labels_dir_dict: Dict[Task, Path],
            splits: Splits) -> None:
        self._features_paths = features_paths
        self._labels_dir_dict = labels_dir_dict
        self._splits = splits

    def provide(self, split_key: str):
        split = self._splits[split_key]
        split_features_paths = [
            features_path for features_path in self._features_paths
            if name_from_path(features_path) in split]
        if not split_features_paths:
            raise ValueError("No feature files from split found in dataset dir")
        split_labels_path_dicts = _create_labels_path_dicts(
            split_features_paths, self._labels_dir_dict)
        return split_features_paths, split_labels_path_dicts


def create_features_paths(features_dir: Path, feature_name: str) -> List[Path]:
    if not features_dir.is_dir():
        raise ValueError(f"Not a valid directory for features: {features_dir}")
    features_paths = sorted(features_dir.glob(f"**/*{feature_name}.npy"))
    if not features_paths:
        raise ValueError(
            f"No features of type {feature_name} found in "
            f"features dir {features_dir}")
    return features_paths


def load_or_create_splits(
        features_paths: List[Path], labels_dir_dict: Dict[Task, Path],
        splits_path: Path) -> Splits:
    if not splits_path.exists():
        _create_splits(features_paths, labels_dir_dict, splits_path)
    return _read_splits(splits_path)


def name_from_path(features_path: Path) -> str:
    return features_path.parent.stem


def _create_splits(
        features_paths: List[Path], labels_dir_dict: Dict[Task, Path],
        splits_path: Path) -> None:
    labels_path_dicts = _create_labels_path_dicts(
        features_paths, labels_dir_dict)
    labels_path_exists = [
        _any_labels_exist(labels_path_dict)
        for labels_path_dict in labels_path_dicts]
    all_names = [
        name_from_path(features_path) for features_path in features_paths]
    unlabeled_split = [
        name for name, labels_path_exists in zip(all_names, labels_path_exists)
        if not labels_path_exists]
    labeled_names = [
        name for name, labels_path_exists in zip(all_names, labels_path_exists)
        if labels_path_exists]
    n_labeled = len(labeled_names)
    # Reset random seed for consistency.
    random.seed()
    shuffled_labeled_names = random.sample(labeled_names, n_labeled)
    n_train = round(FRACTION_TRAIN * n_labeled)
    n_validation = round(FRACTION_VALIDATION * n_labeled)
    validation_end = n_train + n_validation
    train_split = sorted(shuffled_labeled_names[:n_train])
    validation_split = sorted(shuffled_labeled_names[n_train:validation_end])
    test_split = sorted(shuffled_labeled_names[validation_end:])
    splits = {
        SPLIT_KEY_TRAIN: train_split, SPLIT_KEY_VALIDATION: validation_split,
        SPLIT_KEY_TEST: test_split, SPLIT_KEY_UNLABELED: unlabeled_split}
    _write_splits(splits_path, splits)


def _create_labels_path_dicts(
        features_paths: List[Path], labels_dir_dict: Dict[Task, Path]
) -> List[Dict[Task, Path]]:
    labels_path_dicts = [
        _get_labels_path_dict(labels_dir_dict, features_path)
        for features_path in features_paths]
    return labels_path_dicts


def _read_splits(splits_path: Path) -> Splits:
    logging.debug(f"Reading dataset splits file at {splits_path}")
    splits = defaultdict(list)
    with splits_path.open("r") as splits_file:
        reader = csv.DictReader(splits_file)
        for row in reader:
            splits[row[FIELD_SPLIT_KEY]].append(row[FIELD_VIDEO_NAME])
    return splits


def _write_splits(splits_path: Path, splits: Splits) -> None:
    logging.warning(f"Creating dataset splits file at {splits_path}")
    with splits_path.open("w") as splits_file:
        writer = csv.DictWriter(splits_file, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for split_key in splits:
            split = splits[split_key]
            for video_name in split:
                writer.writerow({
                    FIELD_VIDEO_NAME: video_name, FIELD_SPLIT_KEY: split_key})


def _get_labels_path_dict(
        labels_dir_dict: Dict[Task, Path], features_path: Path
) -> Dict[Task, Path]:
    return {
        task: labels_dir / (name_from_path(features_path) + JSON_EXTENSION)
        for task, labels_dir in labels_dir_dict.items()}


def _any_labels_exist(labels_path_dict: Dict[Task, Path]) -> bool:
    for labels_path in labels_path_dict.values():
        if labels_path.exists():
            return True
    return False
