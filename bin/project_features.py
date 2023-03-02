#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging
from pathlib import Path

import numpy as np

from spivak.application.argument_parser import get_args
from spivak.application.dataset_creation import create_label_maps
from spivak.application.feature_utils import make_output_directories, \
    VideoFeatureInfo, create_video_feature_infos
from spivak.application.model_creation import load_projector
from spivak.models.projector import Projector

PROJECTED_FEATURE_NAME = "projected"


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = get_args()
    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    video_feature_infos = create_video_feature_infos(
        [features_dir], [args.feature_name], results_dir,
        PROJECTED_FEATURE_NAME)
    print(f"Found {len(video_feature_infos)} video feature files")
    results_dir.mkdir(parents=True, exist_ok=True)
    make_output_directories(video_feature_infos)
    label_maps = create_label_maps(args)
    projector = load_projector(args, label_maps)
    for video_feature_info in video_feature_infos:
        _create_projected_features_file(video_feature_info, projector)


def _create_projected_features_file(
        video_feature_info: VideoFeatureInfo, projector: Projector) -> None:
    input_path = video_feature_info.input_paths[0]
    features = np.load(str(input_path))
    projected = projector.project(features)
    print(f"Writing projected features to {video_feature_info.output_path}")
    np.save(str(video_feature_info.output_path), projected)


if __name__ == "__main__":
    main()
