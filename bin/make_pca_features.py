#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from spivak.feature_extraction.extraction import PCATransformer, \
    extractor_type_to_feature_name, EXTRACTOR_TYPE_RESNET_TF2

DEFAULT_TAG = "PCA512"
NUMPY_EXTENSION = ".npy"


class Args:
    FEATURES_DIR = "features_dir"
    FEATURES = "features"
    PCA_PATH = "pca"
    TAG = "tag"


def main() -> None:
    args = _get_command_line_arguments()
    pca_transformer = PCATransformer(Path(args[Args.PCA_PATH]))
    feature_name = extractor_type_to_feature_name(args[Args.FEATURES])
    features_dir = Path(args[Args.FEATURES_DIR])
    raw_features_paths = features_dir.glob(
        "**/*" + feature_name + NUMPY_EXTENSION)
    tag = args[Args.TAG]
    for raw_features_path in sorted(raw_features_paths):
        out_path = (raw_features_path.parent /
                    f"{raw_features_path.stem}_{tag}{NUMPY_EXTENSION}")
        print(f"Going to generate {out_path} from {raw_features_path}.")
        raw_features = np.load(str(raw_features_path))
        pca_features = pca_transformer.transform(raw_features)
        np.save(str(out_path), pca_features)


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + Args.FEATURES_DIR, required=True,
        help="Dataset directory for reading and writing features")
    parser.add_argument(
        "--" + Args.PCA_PATH, required=True,
        help="Pickle containing the PCA transform")
    parser.add_argument(
        "--" + Args.TAG, required=False,
        help="Tag to append to output file name", default=DEFAULT_TAG)
    parser.add_argument(
        "--" + Args.FEATURES, required=False,
        help="What type of raw input features to use",
        default=EXTRACTOR_TYPE_RESNET_TF2, choices=[EXTRACTOR_TYPE_RESNET_TF2])
    args_dict = vars(parser.parse_args())
    return args_dict


if __name__ == "__main__":
    main()
