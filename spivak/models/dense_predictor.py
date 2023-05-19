# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tensorflow.keras import Model

from spivak.data.dataset import VideoDatum, TASK_NAMES, Task
from spivak.data.output_names import OUTPUT_DETECTION_SCORE, \
    OUTPUT_DETECTION_SCORE_NMS, OUTPUT_DETECTION_THRESHOLDED, OUTPUT_LABEL, \
    OUTPUT_SEGMENTATION, OUTPUT_TARGET
from spivak.data.soccernet_label_io import \
    segmentation_targets_from_change_labels
from spivak.data.video_chunk_iterator import VideoChunkIteratorProvider
from spivak.models.assembly.head import PredictorHeadInterface
from spivak.models.non_maximum_suppression import FlexibleNonMaximumSuppression
from spivak.models.predictor import PredictorInterface, VideoOutputs

DETECTION_SCORE_THRESHOLD = 0.5
OUTPUT_CONFIDENCE = "confidence"
OUTPUT_DELTA = "delta"
OUTPUT_CONFIDENCE_AUX = "confidence_aux"
OUTPUT_DELTA_AUX = "delta_aux"
PREDICTION_OUTPUTS_TO_SAVE = [
    OUTPUT_DETECTION_SCORE, OUTPUT_DETECTION_SCORE_NMS,
    OUTPUT_DETECTION_THRESHOLDED, OUTPUT_CONFIDENCE, OUTPUT_DELTA,
    OUTPUT_SEGMENTATION]


class DensePredictor(PredictorInterface):

    def __init__(
            self, model: Model, predictor_heads: List[PredictorHeadInterface],
            video_chunk_iterator_provider: VideoChunkIteratorProvider,
            throw_out_delta: bool, profile: bool) -> None:
        self._model = model
        self._predictor_heads = predictor_heads
        self._video_chunk_iterator_provider = video_chunk_iterator_provider
        self._throw_out_delta = throw_out_delta
        self._profile = profile

    def predict_video(self, video_datum: VideoDatum) -> VideoOutputs:
        video_outputs = self.predict_video_base(video_datum.features)
        if OUTPUT_CONFIDENCE in video_outputs:
            video_outputs[OUTPUT_DETECTION_SCORE] = create_detection_scores(
                video_outputs, self._throw_out_delta)
        return video_outputs

    def predict_video_and_save(
            self, video_datum: VideoDatum, nms: FlexibleNonMaximumSuppression,
            base_path: Path) -> None:
        video_outputs = self.predict_video(video_datum)
        DensePredictor.save_predictions(video_outputs, nms, base_path)
        DensePredictor.save_labels(video_datum, base_path)

    def load_weights(self, weights_path: str) -> None:
        self._model.load_weights(weights_path)

    def save_model(self, model_path: str) -> None:
        self._model.save(model_path)

    @staticmethod
    def save_predictions(
            video_predictions: VideoOutputs, nms: FlexibleNonMaximumSuppression,
            base_path: Path) -> None:
        if OUTPUT_DETECTION_SCORE in video_predictions:
            detection_scores = video_predictions[OUTPUT_DETECTION_SCORE]
            # Apply non-maxima suppression if required
            detection_scores_nms = nms.maybe_apply(detection_scores)
            # Get all the detections whose confidence is over the threshold.
            detection_thresholded = np.where(
                detection_scores_nms >= DETECTION_SCORE_THRESHOLD, 1, 0)
            # Store the results
            video_predictions[OUTPUT_DETECTION_SCORE_NMS] = detection_scores_nms
            video_predictions[OUTPUT_DETECTION_THRESHOLDED] = \
                detection_thresholded
        # Save the results to numpy files.
        for prediction_output_name in PREDICTION_OUTPUTS_TO_SAVE:
            if prediction_output_name in video_predictions:
                predictions = video_predictions[prediction_output_name]
                file_path = f"{str(base_path)}_{prediction_output_name}.npy"
                np.save(file_path, predictions)

    @staticmethod
    def save_labels(video_datum: VideoDatum, base_path: Path) -> None:
        # Save available labels.
        for task in video_datum.tasks:
            if video_datum.valid_labels(task):
                task_label_file_name = (
                    f"{str(base_path)}_{TASK_NAMES[task]}_{OUTPUT_LABEL}.npy")
                np.save(task_label_file_name, video_datum.labels(task))
        if video_datum.valid_labels(Task.SEGMENTATION):
            # Also save segmentation targets for visualization.
            change_labels = video_datum.labels(Task.SEGMENTATION)
            segmentation_targets = segmentation_targets_from_change_labels(
                change_labels)
            task_label_file_name = (
                f"{str(base_path)}_{TASK_NAMES[Task.SEGMENTATION]}_"
                f"{OUTPUT_TARGET}.npy")
            np.save(task_label_file_name, segmentation_targets)

    def predict_video_base(self, video_features: np.ndarray) -> VideoOutputs:
        input_chunk_batch, valid_chunk_sizes = self._prepare_input_batch(
            video_features)
        all_head_chunk_outputs = self._predict(
            input_chunk_batch, valid_chunk_sizes)
        return self._accumulate_all_head_chunk_outputs(
            all_head_chunk_outputs, video_features)

    def _prepare_input_batch(
            self, video_features: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        input_chunk_iterator = self._video_chunk_iterator_provider.provide(
            video_features)
        return input_chunk_iterator.prepare_input_batch()

    def _predict(
            self, input_chunk_batch: np.ndarray,
            valid_chunk_sizes: List[int]) -> List[List[np.ndarray]]:
        # predict_on_batch() seems to be safer than predict() relating to memory
        # leak issues: https://github.com/keras-team/keras/issues/13118.
        # Unfortunately, it might require more total GPU memory. In spite of
        # that, we're using it here, since when using predict(), we would run
        # out of memory when running prediction on the Combination x 2 (
        # baidu_2.0) features with TensorFlow 2.7.0.
        if self._profile:
            logging.error(f"input_chunk_batch.shape: {input_chunk_batch.shape}")
        all_head_chunk_output_batch = self._model.predict_on_batch(
            input_chunk_batch)
        if len(self._predictor_heads) == 1:
            all_head_chunk_output_batch = [all_head_chunk_output_batch]
        # Postprocess the result.
        return [
            [
                _post_process(
                    head_chunk_output_batch[c:c + 1], predictor_head,
                    valid_size)
                for c, valid_size in enumerate(valid_chunk_sizes)
            ]
            for predictor_head, head_chunk_output_batch in zip(
                self._predictor_heads, all_head_chunk_output_batch)
        ]

    def _accumulate_all_head_chunk_outputs(
            self, all_head_chunk_outputs: List[List[np.ndarray]],
            video_features: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            predictor_head.name: self._accumulate_head_chunk_outputs(
                head_chunk_outputs, video_features, predictor_head)
            for predictor_head, head_chunk_outputs in zip(
                self._predictor_heads, all_head_chunk_outputs)
        }

    def _accumulate_head_chunk_outputs(
            self, head_chunk_outputs: List[np.ndarray],
            video_features: np.ndarray, predictor_head: PredictorHeadInterface
    ) -> np.ndarray:
        result_chunk_iterator = self._video_chunk_iterator_provider.provide(
            video_features)
        num_frames = video_features.shape[0]
        head_outputs_shape = (
            num_frames, predictor_head.num_classes,
            predictor_head.output_dimension)
        head_outputs = np.zeros(head_outputs_shape)
        result_chunk_iterator.accumulate_chunk_outputs(
            head_outputs, head_chunk_outputs)
        return np.squeeze(head_outputs, axis=2)


def create_detection_scores(
        video_outputs: Dict[str, np.ndarray],
        throw_out_delta: bool) -> np.ndarray:
    confidences = video_outputs[OUTPUT_CONFIDENCE]
    if throw_out_delta or OUTPUT_DELTA not in video_outputs:
        return confidences
    deltas = video_outputs[OUTPUT_DELTA]
    # Put each detection confidence score from detections in the correct
    # place in detection_scores.
    num_frames, num_classes = confidences.shape
    detection_scores = np.zeros((num_frames, num_classes))
    frame_index_matrix = np.empty((num_frames, num_classes), dtype=int)
    # Broadcasts the range on the right into the matrix on the left,
    # repeating the range num_classes times.
    frame_index_matrix[:] = np.arange(num_frames, dtype=int).reshape(
        (num_frames, 1))
    # Figure out what position each delta maps into in detection_scores.
    target_frame_indexes = frame_index_matrix + np.rint(deltas).astype(int)
    # Metrics are better on SoccerNet when keeping the detections that
    # would fall outside the video, but I could make this optional.
    target_frame_indexes = np.clip(target_frame_indexes, 0, num_frames - 1)
    for frame_index in range(num_frames):
        for class_index in range(num_classes):
            score = confidences[frame_index, class_index]
            target_frame_index = target_frame_indexes[frame_index, class_index]
            if score > detection_scores[target_frame_index, class_index]:
                detection_scores[target_frame_index, class_index] = score
    return detection_scores


def _post_process(
        model_output, predictor_head: PredictorHeadInterface, valid_size: int):
    return predictor_head.post_process(model_output)[0:valid_size]
