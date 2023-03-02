#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.decomposition import IncrementalPCA

from spivak.application.feature_utils import read_and_concatenate_features
from spivak.data.dataset_splits import SPLIT_KEY_TRAIN
from spivak.data.soccernet_reader import GamePathsReader
from spivak.feature_extraction.extraction import \
    extractor_type_to_feature_name, EXTRACTOR_TYPE_RESNET_TF2

# Estimate the PCA transform using some SoccerNet data. Save it to a numpy file.
N_COMPONENTS = 512
# In a simple experiment, whitening gave significantly worse results when
# using ResNet features with the context-aware loss.
DEFAULT_WHITEN = False


class Args:
    FEATURES_DIR = "features_dir"
    SPLITS_DIR = "splits_dir"
    OUT_PATH = "out_path"
    FEATURES = "features"
    WHITEN = "whiten"
    NO_WHITEN = "no_whiten"


def main():
    args = _get_command_line_arguments()
    extractor_type = args[Args.FEATURES]
    features_dir = Path(args[Args.FEATURES_DIR])
    splits_dir = Path(args[Args.SPLITS_DIR])
    features = _read_soccernet_features(
        features_dir, splits_dir, extractor_type)
    # Compute the transform.
    whiten = args[Args.WHITEN]
    print("Creating the PCA transform")
    incremental_pca = IncrementalPCA(n_components=N_COMPONENTS, whiten=whiten)
    incremental_pca.fit(features)
    # Write out the PCA transform
    out_path = args[Args.OUT_PATH]
    if not out_path:
        out_path = _create_pca_file_path(
            features_dir, extractor_type, N_COMPONENTS, whiten)
    print(f"Saving result to {out_path}")
    with open(out_path, "wb") as out_file:
        pickle.dump(incremental_pca, out_file)


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + Args.OUT_PATH, help="Optional: output file path", required=False)
    parser.add_argument(
        "--" + Args.FEATURES_DIR, required=True,
        help="Directory in which to store intermediate video features")
    parser.add_argument(
        "--" + Args.FEATURES, required=False,
        help="What type of features to use", default=EXTRACTOR_TYPE_RESNET_TF2,
        choices=[EXTRACTOR_TYPE_RESNET_TF2])
    parser.add_argument(
        "--" + Args.SPLITS_DIR, required=True, type=str,
        help="Directory containing splits information")
    parser.add_argument(
        "--" + Args.WHITEN, help="Whiten the transformation", required=False,
        action='store_true', dest=Args.WHITEN)
    parser.add_argument(
        "--" + Args.NO_WHITEN, help="Don't whiten the transformation",
        required=False, action='store_false', dest=Args.WHITEN)
    parser.set_defaults(**{Args.WHITEN: DEFAULT_WHITEN})
    args_dict = vars(parser.parse_args())
    return args_dict


def _read_soccernet_features(
        features_dir: Path, splits_dir: Path,
        extractor_type: str) -> np.ndarray:
    game_list = GamePathsReader.read_game_list_v2(splits_dir, SPLIT_KEY_TRAIN)
    feature_name = extractor_type_to_feature_name(extractor_type)
    return read_and_concatenate_features(features_dir, game_list, feature_name)


def _create_pca_file_path(
        features_dir: Path, extractor_type: str, n_components: int,
        whiten: bool) -> str:
    return str(features_dir /
               _create_pca_file_name(extractor_type, n_components, whiten))


def _create_pca_file_name(
        extractor_type: str, n_components: int, whiten: bool) -> str:
    return f"pca_transform_{extractor_type}_{n_components}_whiten_{whiten}.pkl"


if __name__ == "__main__":
    main()
