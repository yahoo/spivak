#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from scipy.interpolate import interp1d

from spivak.application.feature_utils import make_output_directories, \
    VideoFeatureInfo, create_video_feature_infos
from spivak.models.delta_dense_predictor import clip_frames

RESAMPLING_ZEROS = "zeros"
RESAMPLING_REPEAT = "repeat"
RESAMPLING_INTERPOLATE = "interpolate"
RESAMPLING_MAX = "max"
IDENTITY_NORMALIZER = "identity"


class Args:
    INPUT_DIRS = "input_dirs"
    INPUT_FEATURE_NAMES = "input_feature_names"
    OUTPUT_DIR = "output_dir"
    OUTPUT_FEATURE_NAME = "output_feature_name"
    FACTORS = "factors"
    NORMALIZERS = "normalizers"
    RESAMPLING = "resampling"


def main() -> None:
    args = _get_command_line_arguments()
    input_dirs = [Path(p) for p in args[Args.INPUT_DIRS]]
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise ValueError(f"Input directory failed is_dir(): {input_dir}")
    input_feature_names = args[Args.INPUT_FEATURE_NAMES]
    normalizers = _read_normalizers(
        args[Args.NORMALIZERS], len(input_feature_names))
    factors = args[Args.FACTORS]
    resampling = args[Args.RESAMPLING]
    output_feature_name = args[Args.OUTPUT_FEATURE_NAME]
    output_dir = Path(args[Args.OUTPUT_DIR])
    output_dir.mkdir(parents=True, exist_ok=True)
    video_feature_infos = create_video_feature_infos(
        input_dirs, input_feature_names, output_dir, output_feature_name)
    print(f"Found {len(video_feature_infos)} video feature files")
    make_output_directories(video_feature_infos)
    for video_feature_info in video_feature_infos:
        _transform_features_file(
            video_feature_info, normalizers, factors, resampling)


def _read_normalizers(
        normalizers_arg: Optional[List[str]], n_features: int) -> List:
    if normalizers_arg is None:
        return [None] * n_features
    assert len(normalizers_arg) == n_features
    return [
        _read_normalizer(normalizer_arg) for normalizer_arg in normalizers_arg]


def _read_normalizer(normalizer_arg: str) -> Any:
    if normalizer_arg == IDENTITY_NORMALIZER:
        return None
    with Path(normalizer_arg).open("rb") as pickle_file:
        return pickle.load(pickle_file)


def _transform_features_file(
        video_feature_info: VideoFeatureInfo, normalizers: List[Any],
        factors: List[float], resampling: str) -> None:
    # Read the features from video_feature_info.input_paths, resample them using
    # the specified factors and the provided resampling strategy, concatenate
    # the results, then save to numpy output file
    # video_feature_info.output_path.
    transformed_list = []
    for input_path, normalizer, factor in zip(
            video_feature_info.input_paths, normalizers, factors):
        original = np.load(str(input_path))
        if normalizer:
            features_to_resample = normalizer.transform(original)
        else:
            features_to_resample = original
        transformed = _resample_features(
            features_to_resample, factor, resampling)
        transformed_list.append(transformed)
    min_num_frames = min(
        transformed.shape[0] for transformed in transformed_list)
    transformed_clipped_list = [
        clip_frames(transformed, min_num_frames)
        for transformed in transformed_list]
    transformed_features = np.concatenate(transformed_clipped_list, axis=1)
    print(f"Writing transformed features to {video_feature_info.output_path}")
    np.save(str(video_feature_info.output_path), transformed_features)


def _resample_features(
        original: np.ndarray, factor: float, resampling: str) -> np.ndarray:
    if factor == 1.0:
        return original
    if resampling == RESAMPLING_INTERPOLATE:
        n_original_times = len(original)
        # We don't know the actual timestamps or frequency of the original
        # features, so we just set the time step to 1.0 (starting at 1.0),
        # since what really matters are the relative time values.
        original_times = np.linspace(1.0, n_original_times, n_original_times)
        interpolation = interp1d(
            original_times, original, axis=0, copy=False, kind="linear",
            bounds_error=False, fill_value="extrapolate")
        desired_times = np.linspace(
            1.0 / factor, n_original_times, round(factor * n_original_times))
        transformed = interpolation(desired_times)
    elif resampling == RESAMPLING_ZEROS:
        raise NotImplementedError()
    elif resampling == RESAMPLING_REPEAT:
        raise NotImplementedError()
    elif resampling == RESAMPLING_MAX:
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown resampling: {resampling}")
    return transformed


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + Args.INPUT_DIRS,
        help='One or more input directories containing features',
        nargs="+", required=True, type=str)
    parser.add_argument(
        "--" + Args.INPUT_FEATURE_NAMES,
        help='One or more feature file name endings (e.g. ResNET_TF2, without '
             'the .npy part)', nargs="+", required=True, type=str)
    parser.add_argument(
        "--" + Args.OUTPUT_DIR, required=True,
        help="Directory for the output features", type=str)
    parser.add_argument(
        "--" + Args.OUTPUT_FEATURE_NAME, required=True,
        help="Name of the output feature for feature file name endings "
             "(e.g. ResNET_TF2, without the .npy part)", type=str)
    parser.add_argument(
        "--" + Args.NORMALIZERS, required=False, help=(
            "One or more pickle files containing normalizers for the features. "
            f"Use \"{IDENTITY_NORMALIZER}\" to apply no normalization to "
            f"the corresponding feature. Normalization is applied before "
            f"resampling."), type=str, nargs="+")
    parser.add_argument(
        "--" + Args.FACTORS, required=True,
        help="One or more factors (floats) for the new sampling rates",
        type=float, nargs="+")
    parser.add_argument(
        "--" + Args.RESAMPLING, required=True,
        help="How to resample the features to achieve higher sampling rates",
        choices=[
            RESAMPLING_ZEROS, RESAMPLING_INTERPOLATE, RESAMPLING_REPEAT,
            RESAMPLING_MAX], type=str)
    args_dict = vars(parser.parse_args())
    return args_dict


if __name__ == "__main__":
    main()
