# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.decomposition import IncrementalPCA

from spivak.feature_extraction.soccernet_v2 import FeatureExtractorResNetTF2, \
    SoccerNetPCATransformer, PCAInterface, RawFeatureExtractorInterface

# Feature extraction constants.
DEFAULT_TIME_STRIDE = 0.5
GAME_START_TIME = 0
GAME_END_TIME = 0
EXTRACTOR_TYPE_RESNET_TF2 = "ResNet_TF2"
# SOCCERNET_FEATURE_NAME_AAA have to match convention in SoccerNet codebase.
SOCCERNET_FEATURE_NAME_RESNET_TF2 = "ResNET_TF2"
MODEL_WEIGHTS_RESNET_TF2 = "resnet152_weights_tf_dim_ordering_tf_kernels.h5"
PCA_RESNET_TF2 = "pca_512_TF2.pkl"
PCA_SCALER_RESNET_TF2 = "average_512_TF2.pkl"


class VideoInfo:

    def __init__(
            self, video_path: Path, features_dir: Path,
            relative_features_dir: Path) -> None:
        self.video_path = video_path
        self.features_dir = features_dir
        self.relative_features_dir = relative_features_dir


class PCATransformer(PCAInterface):

    def __init__(self, file_path: Path) -> None:
        with file_path.open("rb") as pickle_file:
            self.projection: IncrementalPCA = pickle.load(pickle_file)

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        return self.projection.transform(raw_features)


class FeatureExtractor:

    def __init__(
            self, raw_feature_extractor: RawFeatureExtractorInterface,
            feature_file_prefix: str, pca_transformer: PCAInterface) -> None:
        self.raw_feature_extractor = raw_feature_extractor
        self.feature_file_prefix = feature_file_prefix
        self.pca_transformer = pca_transformer

    def make_features(self, video_info: VideoInfo) -> None:
        features_file_name = f"{self.feature_file_prefix}_PCA512.npy"
        features_path = video_info.features_dir / features_file_name
        if features_path.exists():
            logging.warning(
                f"*** Skipping feature creation (already exists) for: "
                f"{features_path}")
            return
        logging.info(f"Creating feature file: {features_path}")
        features_array = self._create_features(video_info)
        np.save(str(features_path), features_array)

    def _create_features(self, video_info: VideoInfo) -> np.ndarray:
        raw_features = self._get_raw_features(video_info)
        return self.pca_transformer.transform(raw_features)

    def _get_raw_features(self, video_info: VideoInfo) -> np.ndarray:
        raw_file_name = f"{self.feature_file_prefix}.npy"
        raw_path = video_info.features_dir / raw_file_name
        if raw_path.exists():
            logging.warning(
                f"*** Skipping RAW features creation (already exists) for: "
                f"{raw_path}")
            return np.load(str(raw_path))
        logging.info(f"Creating RAW feature file: {raw_path}")
        raw_array = self.raw_feature_extractor.extract_features(
            str(video_info.video_path), GAME_START_TIME, GAME_END_TIME)
        np_raw_features = np.array(raw_array)
        np.save(str(raw_path), np_raw_features)
        return np_raw_features


def extract_features_from_videos(
        video_paths: List[Path], features_dir: Path,
        feature_extractor: FeatureExtractor) -> None:
    video_infos = _create_video_infos(video_paths, features_dir)
    _make_feature_directories(video_infos)
    _make_features(video_infos, feature_extractor)


def create_feature_extractor(
        extractor_type: str, model_dir: Path) -> FeatureExtractor:
    if extractor_type == EXTRACTOR_TYPE_RESNET_TF2:
        fps = 1.0 / DEFAULT_TIME_STRIDE
        resnet_tf2_path = str(model_dir / MODEL_WEIGHTS_RESNET_TF2)
        soccernet_feature_extractor = FeatureExtractorResNetTF2(
            resnet_tf2_path, fps=fps)
        pca_path = model_dir / PCA_RESNET_TF2
        pca_scaler_path = model_dir / PCA_SCALER_RESNET_TF2
        pca_transformer = SoccerNetPCATransformer(pca_path, pca_scaler_path)
    else:
        raise ValueError(f"Invalid value for extractor type: {extractor_type}")
    feature_file_prefix = extractor_type_to_feature_name(extractor_type)
    return FeatureExtractor(
        soccernet_feature_extractor, feature_file_prefix, pca_transformer)


def extractor_type_to_feature_name(extractor_type: str) -> str:
    if extractor_type == EXTRACTOR_TYPE_RESNET_TF2:
        soccernet_feature_name = SOCCERNET_FEATURE_NAME_RESNET_TF2
    else:
        raise ValueError(f"Invalid value for extractor type: {extractor_type}")
    return soccernet_feature_name


def _create_video_infos(
        video_paths: List[Path], features_dir: Path) -> List[VideoInfo]:
    return [_create_video_info(video_path, features_dir) for video_path in
            video_paths]


def _create_video_info(video_path: Path, features_dir: Path) -> VideoInfo:
    video_features_dir = features_dir / video_path.stem
    relative_features_dir = video_features_dir.relative_to(features_dir)
    return VideoInfo(video_path, video_features_dir, relative_features_dir)


def _make_feature_directories(video_infos: List[VideoInfo]) -> None:
    for video_info in video_infos:
        video_info.features_dir.mkdir(exist_ok=True)


def _make_features(
        video_infos: List[VideoInfo],
        feature_extractor: FeatureExtractor) -> None:
    for video_info in video_infos:
        feature_extractor.make_features(video_info)
