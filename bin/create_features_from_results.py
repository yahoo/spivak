#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from spivak.application.feature_utils import make_output_directories, \
    VideoFeatureInfo, create_video_feature_infos
from spivak.models.delta_dense_predictor import clip_frames
from spivak.models.dense_predictor import OUTPUT_DELTA, OUTPUT_CONFIDENCE


class Args:
    INPUT_DIRS = "input_dirs"
    OUTPUT_NAME = "output_name"
    OUTPUT_DIR = "output_dir"


def main() -> None:
    args = _get_command_line_arguments()
    input_dirs = [Path(p) for p in args[Args.INPUT_DIRS]]
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise ValueError(f"Input directory failed is_dir(): {input_dir}")
    output_name = args[Args.OUTPUT_NAME]
    output_dir = Path(args[Args.OUTPUT_DIR])
    video_infos = create_video_feature_infos(
        input_dirs, [output_name] * len(input_dirs), output_dir, output_name)
    print(f"Found {len(video_infos)} video result files")
    output_dir.mkdir(parents=True, exist_ok=True)
    make_output_directories(video_infos)
    for video_info in video_infos:
        _create_combined_file(video_info)


def _create_combined_file(video_info: VideoFeatureInfo) -> None:
    results_list = []
    for input_path in video_info.input_paths:
        input_results = np.load(str(input_path))
        results_list.append(input_results)
    min_num_frames = min(results.shape[0] for results in results_list)
    clipped_results_list = [
        clip_frames(results, min_num_frames) for results in results_list]
    combined_results = np.stack(clipped_results_list, axis=2)
    print(f"Writing combined results to {video_info.output_path}")
    np.save(str(video_info.output_path), combined_results)


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + Args.INPUT_DIRS,
        help="One or more input directories containing results",
        nargs="+", required=True, type=str)
    parser.add_argument(
        "--" + Args.OUTPUT_NAME,
        help="Which output type to read and write", required=True, type=str,
        choices=[OUTPUT_CONFIDENCE, OUTPUT_DELTA])
    parser.add_argument(
        "--" + Args.OUTPUT_DIR, required=True,
        help="Directory for the output features", type=str)
    args_dict = vars(parser.parse_args())
    return args_dict


if __name__ == "__main__":
    main()
