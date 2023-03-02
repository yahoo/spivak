# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import pickle
from pathlib import Path
from typing import Union

import numpy as np
from scipy.special import logit, expit

from spivak.data.dataset import VideoDatum
from spivak.data.output_names import OUTPUT_DETECTION_SCORE
from spivak.models.delta_dense_predictor import DeltaDensePredictor, clip_frames
from spivak.models.dense_predictor import DensePredictor, \
    create_detection_scores, OUTPUT_CONFIDENCE, OUTPUT_DELTA
from spivak.models.non_maximum_suppression import FlexibleNonMaximumSuppression
from spivak.models.predictor import PredictorInterface, VideoOutputs

AveragingPredictor = Union[
    "ConfidenceAveragingPredictor", "DeltaAveragingPredictor"]


class ConfidenceAveragingPredictor(PredictorInterface):

    def __init__(self, weights: np.ndarray, use_logits: bool) -> None:
        self.weights = weights
        self._use_logits = use_logits

    def predict_video(self, video_datum: VideoDatum) -> VideoOutputs:
        if self._use_logits:
            confidence = expit(np.average(
                logit(video_datum.features), axis=2, weights=self.weights))
        else:
            confidence = np.average(
                video_datum.features, axis=2, weights=self.weights)
        return {
            OUTPUT_CONFIDENCE: confidence, OUTPUT_DETECTION_SCORE: confidence}

    def predict_video_and_save(
            self, video_datum: VideoDatum, nms: FlexibleNonMaximumSuppression,
            base_path: Path) -> None:
        video_outputs = self.predict_video(video_datum)
        DensePredictor.save_predictions(video_outputs, nms, base_path)
        DensePredictor.save_labels(video_datum, base_path)

    def save_model(self, model_path: str) -> None:
        with open(model_path, "wb") as model_file:
            pickle.dump(self, model_file)

    def load_weights(self, weights_path: str) -> None:
        raise NotImplementedError()


class DeltaAveragingPredictor(PredictorInterface):

    def __init__(
            self, weights: np.ndarray, confidence_dir: Path,
            use_arcs: bool, delta_radius: float) -> None:
        self.weights = weights
        self._confidence_dir = confidence_dir
        self._use_arcs = use_arcs
        self._delta_radius = delta_radius

    def predict_video(self, video_datum: VideoDatum) -> VideoOutputs:
        if self._use_arcs:
            delta = self._delta_radius * np.tanh(np.average(
                np.arctanh(video_datum.features / self._delta_radius),
                axis=2, weights=self.weights
            ))
        else:
            delta = np.average(
                video_datum.features, axis=2, weights=self.weights)
        confidence_path = DeltaDensePredictor.confidence_path(
            self._confidence_dir, video_datum.relative_path)
        confidence = np.load(str(confidence_path))
        # If delta and confidence are generated using different types of
        # features, they might have small size differences.
        min_num_frames = min(delta.shape[0], confidence.shape[0])
        confidence = clip_frames(confidence, min_num_frames)
        delta = clip_frames(delta, min_num_frames)
        video_outputs = {OUTPUT_CONFIDENCE: confidence, OUTPUT_DELTA: delta}
        video_outputs[OUTPUT_DETECTION_SCORE] = create_detection_scores(
            video_outputs, throw_out_delta=False)
        return video_outputs

    def predict_video_and_save(
            self, video_datum: VideoDatum, nms: FlexibleNonMaximumSuppression,
            base_path: Path) -> None:
        video_outputs = self.predict_video(video_datum)
        DensePredictor.save_predictions(video_outputs, nms, base_path)
        DensePredictor.save_labels(video_datum, base_path)

    def save_model(self, model_path: str) -> None:
        with open(model_path, "wb") as model_file:
            pickle.dump(self, model_file)

    def load_weights(self, weights_path: str) -> None:
        raise NotImplementedError()
