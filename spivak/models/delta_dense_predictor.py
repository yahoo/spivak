# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from pathlib import Path

import numpy as np

from spivak.data.dataset import VideoDatum
from spivak.data.output_names import OUTPUT_DETECTION_SCORE
from spivak.models.dense_predictor import DensePredictor, \
    create_detection_scores, OUTPUT_CONFIDENCE
from spivak.models.non_maximum_suppression import FlexibleNonMaximumSuppression
from spivak.models.predictor import PredictorInterface, VideoOutputs


class DeltaDensePredictor(PredictorInterface):

    def __init__(
            self, dense_predictor: DensePredictor,
            confidence_dir: Path) -> None:
        self._dense_predictor = dense_predictor
        self._confidence_dir = confidence_dir

    def predict_video(self, video_datum: VideoDatum) -> VideoOutputs:
        video_outputs = self._dense_predictor.predict_video_base(
            video_datum.features)
        confidence_path = DeltaDensePredictor.confidence_path(
            self._confidence_dir, video_datum.relative_path)
        video_outputs[OUTPUT_CONFIDENCE] = np.load(str(confidence_path))
        # Fix possible mismatch in number of frames.
        video_outputs = _fix_video_outputs_frames(video_outputs)
        video_outputs[OUTPUT_DETECTION_SCORE] = create_detection_scores(
            video_outputs, throw_out_delta=False)
        return video_outputs

    def predict_video_and_save(
            self, video_datum: VideoDatum, nms: FlexibleNonMaximumSuppression,
            base_path: Path) -> None:
        video_outputs = self.predict_video(video_datum)
        DensePredictor.save_predictions(video_outputs, nms, base_path)
        DensePredictor.save_labels(video_datum, base_path)

    def load_weights(self, weights_path: str) -> None:
        self._dense_predictor.load_weights(weights_path)

    def save_model(self, model_path: str) -> None:
        self._dense_predictor.save_model(model_path)

    @staticmethod
    def confidence_path(confidence_dir: Path, relative_path: Path) -> Path:
        base_path = confidence_dir / relative_path
        return base_path.parent / f"{base_path.stem}_{OUTPUT_CONFIDENCE}.npy"


def clip_frames(unclipped: np.ndarray, clipped_num_frames: int) -> np.ndarray:
    unclipped_num_frames = unclipped.shape[0]
    if unclipped_num_frames == clipped_num_frames:
        return unclipped
    assert clipped_num_frames == unclipped_num_frames - 1
    return unclipped[:clipped_num_frames]


def _fix_video_outputs_frames(video_outputs: VideoOutputs) -> VideoOutputs:
    min_num_frames = min(output.shape[0] for output in video_outputs.values())
    return {
        output_name: clip_frames(output, min_num_frames)
        for output_name, output in video_outputs.items()}
