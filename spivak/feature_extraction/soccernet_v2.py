# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.
#
# This file incorporates work covered by the following copyright and permission
# notice:
#   Copyright (c) 2021 Silvio Giancola
#   Licensed under the terms of the MIT license.
#   You may obtain a copy of the MIT License at https://opensource.org/licenses/MIT

# This file contains pieces of code taken from the following file.
# https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/20f2f74007c82b68a73c519dff852188df4a8b5a/Features/VideoFeatureExtractor.py
# At Yahoo Inc., the original code was modified and new code was added. The
# code now uses different versions of FrameCV and Frame (for decoding videos),
# which are defined here in SoccerNetDataLoader.py.

import logging
import pickle
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet import preprocess_input

from spivak.feature_extraction.SoccerNetDataLoader import FrameCV, Frame


class RawFeatureExtractorInterface(metaclass=ABCMeta):

    @abstractmethod
    def extract_features(
            self, video_path: str, game_start_time: int, game_end_time: int
    ) -> np.ndarray:
        pass


class PCAInterface(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        pass


class FeatureExtractorResNetTF2(RawFeatureExtractorInterface):

    def __init__(
            self, model_weights_path: str, grabber="opencv", fps=2.0,
            image_transform="crop") -> None:
        self.grabber = grabber
        self.fps = fps
        self.image_transform = image_transform
        base_model = keras.applications.resnet.ResNet152(
            include_top=True, weights=model_weights_path,
            input_tensor=None, input_shape=None, pooling=None, classes=1000)
        # define model with output after polling layer (dim=2048)
        self.model = Model(
            base_model.input, outputs=[base_model.get_layer("avg_pool").output])
        self.model.trainable = False

    def extract_features(
            self, video_path: str, game_start_time: int, game_end_time: int
    ) -> np.ndarray:
        start = None
        video_duration = None
        if game_start_time:
            start = game_start_time
        if game_end_time:
            video_duration = game_end_time - game_start_time
        if self.grabber == "skvideo":
            video_loader = Frame(
                video_path, FPS=self.fps, transform=self.image_transform,
                start=start, duration=video_duration)
        elif self.grabber == "opencv":
            video_loader = FrameCV(
                video_path, FPS=self.fps, transform=self.image_transform,
                start=start, duration=video_duration)
        else:
            raise ValueError(f"Unknown frame grabber: {self.grabber}")
        frames = preprocess_input(video_loader.frames)
        if video_duration is None:
            video_duration = video_loader.time_second
        logging.info(
            f"frames {frames.shape}, fps={frames.shape[0] / video_duration}")
        # predict the features from the frames (adjust batch size for smaller
        # GPU)
        prediction_start_time = time.time()
        features = self.model.predict(frames, batch_size=64, verbose=1)
        prediction_time = time.time() - prediction_start_time
        logging.info(f"feature model prediction time: {prediction_time}")
        logging.info(
            f"features {features.shape}, fps="
            f"{features.shape[0] / video_duration}")
        return features


class SoccerNetPCATransformer(PCAInterface):

    def __init__(self, pca_path: Path, scalar_path: Path) -> None:
        with pca_path.open("rb") as pca_file:
            self.pca = pickle.load(pca_file)
        with scalar_path.open("rb") as scalar_file:
            self.average = pickle.load(scalar_file)

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = features - self.average
        features = self.pca.transform(features)
        return features
