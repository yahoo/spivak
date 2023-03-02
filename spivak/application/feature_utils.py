# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import itertools
from pathlib import Path
from typing import List

import numpy as np


class VideoFeatureInfo:

    def __init__(self, input_paths: List[Path], output_path: Path) -> None:
        self.input_paths = input_paths
        self.output_path = output_path


def make_output_directories(
        video_feature_infos: List[VideoFeatureInfo]) -> None:
    for video_feature_info in video_feature_infos:
        video_feature_info.output_path.parent.mkdir(exist_ok=True, parents=True)


def create_video_feature_infos(
        input_dirs: List[Path], input_feature_names: List[str],
        output_dir: Path, feature_name: str) -> List[VideoFeatureInfo]:
    all_features_input_paths = []
    for current_input_dir, current_input_feature_name in zip(
            input_dirs, input_feature_names):
        current_input_paths = sorted(
            current_input_dir.glob(f"**/*{current_input_feature_name}.npy"))
        all_features_input_paths.append(current_input_paths)
    video_feature_infos = []
    for video_index, _ in enumerate(all_features_input_paths[0]):
        first_input_path = all_features_input_paths[0][video_index]
        relative_dir = first_input_path.parent.relative_to(input_dirs[0])
        video_feature_input_paths = [
            current_input_paths[video_index]
            for current_input_paths in all_features_input_paths]
        video_feature_info = _create_video_feature_info(
            video_feature_input_paths, relative_dir, output_dir, feature_name)
        video_feature_infos.append(video_feature_info)
    return video_feature_infos


def read_and_concatenate_features(
        features_dir: Path, game_list: List[Path],
        feature_name: str) -> np.ndarray:
    features_iterator = itertools.chain.from_iterable(
        _read_game_features(features_dir / game, feature_name)
        for game in game_list)
    return np.concatenate(list(features_iterator))


def _create_video_feature_info(
        input_paths: List[Path], relative_dir: Path, output_dir: Path,
        feature_name: str) -> VideoFeatureInfo:
    game_half = input_paths[0].stem[0]
    output_path = output_dir / relative_dir / f"{game_half}_{feature_name}.npy"
    return VideoFeatureInfo(input_paths, output_path)


def _read_game_features(
        game_dir: Path, feature_name: str) -> List[np.ndarray]:
    if not game_dir.is_dir():
        raise ValueError(f"Could not find game directory: {game_dir}")
    print(f"Reading features from {game_dir}")
    first = np.load(str(game_dir / f"1_{feature_name}.npy"))
    second = np.load(str(game_dir / f"2_{feature_name}.npy"))
    return [first, second]
