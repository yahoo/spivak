#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
import logging
from pathlib import Path
from typing import Dict

from spivak.data.video_io import list_video_paths
from spivak.feature_extraction.extraction import extract_features_from_videos, \
    create_feature_extractor, EXTRACTOR_TYPE_RESNET_TF2


class Args:
    INPUT_VIDEOS_DIR = "input_dir"
    FEATURES_DIR = "features_dir"
    FEATURES_MODELS_DIR = "features_models_dir"
    FEATURES = "features"


def main() -> None:
    args = _get_command_line_arguments()
    logging.getLogger().setLevel(logging.DEBUG)
    input_dir = Path(args[Args.INPUT_VIDEOS_DIR])
    if not input_dir.is_dir():
        raise ValueError(f"Input directory failed is_dir(): {input_dir}")
    features_dir = Path(args[Args.FEATURES_DIR])
    features_dir.mkdir(parents=True, exist_ok=True)
    feature_extractor = create_feature_extractor(
        args[Args.FEATURES], Path(args[Args.FEATURES_MODELS_DIR]))
    video_paths = list_video_paths(input_dir)
    extract_features_from_videos(video_paths, features_dir, feature_extractor)


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + Args.INPUT_VIDEOS_DIR, help="Input directory containing videos",
        required=True)
    parser.add_argument(
        "--" + Args.FEATURES_DIR, required=True,
        help="Directory in which to store intermediate video features")
    parser.add_argument(
        "--" + Args.FEATURES_MODELS_DIR, required=True,
        help="Directory containing models used for extracting video features")
    parser.add_argument(
        "--" + Args.FEATURES, required=False,
        help="What type of features to use", default=EXTRACTOR_TYPE_RESNET_TF2,
        choices=[EXTRACTOR_TYPE_RESNET_TF2])
    args_dict = vars(parser.parse_args())
    return args_dict


if __name__ == "__main__":
    main()
