#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn import preprocessing

from spivak.application.feature_utils import read_and_concatenate_features
from spivak.data.dataset_splits import SPLIT_KEY_TRAIN
from spivak.data.soccernet_reader import GamePathsReader

NORMALIZER_MIN_MAX = "min_max"
NORMALIZER_STANDARD = "standard"
NORMALIZER_STANDARD_NO_MEAN = "standard_no_mean"
NORMALIZER_MAX_ABS = "max_abs"


class Args:
    FEATURES_DIR = "features_dir"
    SPLITS_DIR = "splits_dir"
    NORMALIZER = "normalizer"
    FEATURE_NAME = "feature_name"
    OUT_PATH = "out_path"


def main():
    args = _get_command_line_arguments()
    feature_name = args[Args.FEATURE_NAME]
    normalizer_type = args[Args.NORMALIZER]
    features_dir = Path(args[Args.FEATURES_DIR])
    splits_dir = Path(args[Args.SPLITS_DIR])
    print("Reading all the features")
    features = _read_features(features_dir, splits_dir, feature_name)
    # Create the normalizer and fit it to the features.
    print("Creating the normalizer")
    if normalizer_type == NORMALIZER_MIN_MAX:
        normalizer = preprocessing.MinMaxScaler()
    elif normalizer_type == NORMALIZER_STANDARD:
        normalizer = preprocessing.StandardScaler()
    elif normalizer_type == NORMALIZER_STANDARD_NO_MEAN:
        normalizer = preprocessing.StandardScaler(with_mean=False)
    elif normalizer_type == NORMALIZER_MAX_ABS:
        normalizer = preprocessing.MaxAbsScaler()
    else:
        raise ValueError(f"Unknown normalizer type {normalizer_type}")
    normalizer.fit(features)
    # Write out the normalizer.
    out_path = Path(args[Args.OUT_PATH])
    print(f"Saving normalizer to {out_path}")
    with out_path.open("wb") as out_file:
        pickle.dump(normalizer, out_file)


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + Args.OUT_PATH, required=True, type=str,
        help="Output pickle file path")
    parser.add_argument(
        "--" + Args.FEATURES_DIR, required=True, type=str,
        help="Directory from which to read the video features")
    parser.add_argument(
        "--" + Args.SPLITS_DIR, required=True, type=str,
        help="Directory containing splits information")
    parser.add_argument(
        "--" + Args.FEATURE_NAME, required=True, type=str,
        help="What type of features to read")
    parser.add_argument(
        "--" + Args.NORMALIZER, required=True, type=str,
        choices=[NORMALIZER_STANDARD, NORMALIZER_STANDARD_NO_MEAN,
                 NORMALIZER_MIN_MAX, NORMALIZER_MAX_ABS],
        help="Type of the normalizer")
    args_dict = vars(parser.parse_args())
    return args_dict


def _read_features(
        features_dir: Path, splits_dir: Path, feature_name: str) -> np.ndarray:
    # Get the list of games from the standard training split. Don't need to
    # involve validation data, since the training features should be enough to
    # give us good statistics.
    game_list = GamePathsReader.read_game_list_v2(splits_dir, SPLIT_KEY_TRAIN)
    return read_and_concatenate_features(features_dir, game_list, feature_name)


if __name__ == "__main__":
    main()
