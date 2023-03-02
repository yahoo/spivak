#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging
from pathlib import Path
from typing import List

import numpy as np

from spivak.application.argument_parser import get_args, \
    DETECTOR_AVERAGING_CONFIDENCE, DETECTOR_AVERAGING_DELTA, dir_str_to_path, \
    SharedArgs
from spivak.application.dataset_creation import create_label_maps, \
    create_soccernet_video_data_reader
from spivak.application.model_creation import create_flexible_nms, \
    create_delta_radius
from spivak.application.validation import create_all_video_outputs, \
    create_detections_and_targets
from spivak.data.dataset import Task, VideoDatum, read_numpy_shape
from spivak.data.dataset_splits import SPLIT_KEY_VALIDATION
from spivak.data.label_map import LabelMap
from spivak.evaluation.spotting_evaluation import run_spotting_evaluation, \
    TolerancesConfig, read_tolerances_config
from spivak.models.averaging_predictor import ConfidenceAveragingPredictor, \
    DeltaAveragingPredictor, AveragingPredictor
from spivak.models.non_maximum_suppression import FlexibleNonMaximumSuppression

DELTA_WEIGHT_RANGE = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.0]
CONFIDENCE_LOGIT_WEIGHT_RANGE = [
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
    0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
CONFIDENCE_USE_LOGITS = True
DELTA_USE_ARCS = False


def main() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    args = get_args()
    label_maps = create_label_maps(args)
    soccernet_video_data_reader = create_soccernet_video_data_reader(
        args, label_maps)
    video_data = soccernet_video_data_reader.read(SPLIT_KEY_VALIDATION)
    num_features = _read_num_features(video_data[0]._features_path)
    predictor = _create_averaging_predictor(args, num_features)
    _optimize_predictor(args, predictor, video_data, label_maps[Task.SPOTTING])
    predictor.save_model(args.model)


def _read_num_features(features_path: Path) -> int:
    shape = read_numpy_shape(features_path)
    return shape[2]


def _create_averaging_predictor(
        args: SharedArgs, num_features: int) -> AveragingPredictor:
    weights = np.zeros(num_features)
    if args.detector == DETECTOR_AVERAGING_CONFIDENCE:
        predictor = ConfidenceAveragingPredictor(weights, CONFIDENCE_USE_LOGITS)
    elif args.detector == DETECTOR_AVERAGING_DELTA:
        predictor = DeltaAveragingPredictor(
            weights, dir_str_to_path(args.results_dir), DELTA_USE_ARCS,
            create_delta_radius(args))
    else:
        raise ValueError(f"Unknown averaging predictor type: {args.detector}")
    return predictor


def _optimize_predictor(
        args: SharedArgs, predictor: AveragingPredictor,
        video_data: List[VideoDatum], label_map: LabelMap) -> None:
    weight_range = _create_weight_range(args.detector)
    flexible_nms = create_flexible_nms(args, label_map)
    # Prepare deltas for evaluation
    config_dir = dir_str_to_path(args.config_dir)
    tolerances_config = read_tolerances_config(config_dir)
    logging.info(f"Computing metrics for weights in {weight_range}")
    metrics = [
        _compute_main_metric(
            args, video_data, predictor, weight, flexible_nms,
            tolerances_config, label_map)
        for weight in weight_range
    ]
    logging.info(f"Metrics found: {metrics}")
    max_metric_index = np.argmax(metrics)
    best_weight = weight_range[max_metric_index]
    predictor.weights = _weights_from_weight(best_weight)


def _compute_main_metric(
        args: SharedArgs, video_data: List[VideoDatum],
        predictor: AveragingPredictor, weight: float,
        flexible_nms: FlexibleNonMaximumSuppression,
        tolerances_config: TolerancesConfig, label_map: LabelMap) -> float:
    predictor.weights = _weights_from_weight(weight)
    all_video_outputs = create_all_video_outputs(video_data, predictor)
    detections, targets = create_detections_and_targets(
        video_data, all_video_outputs, flexible_nms)
    spotting_evaluation = run_spotting_evaluation(
        detections, targets, tolerances_config, args.frame_rate,
        label_map.num_classes(), bool(args.prune_classes),
        create_confusion_data_frame=False, label_map=label_map)
    main_metric = spotting_evaluation.average_map_dict[
        spotting_evaluation.main_tolerances_name]
    logging.info(
        f"Got {spotting_evaluation.main_tolerances_name} average-mAP: "
        f"{main_metric} for weight {weight}.")
    return main_metric


def _create_weight_range(detector: str) -> List[float]:
    if detector == DETECTOR_AVERAGING_CONFIDENCE and CONFIDENCE_USE_LOGITS:
        return CONFIDENCE_LOGIT_WEIGHT_RANGE
    elif detector == DETECTOR_AVERAGING_DELTA:
        return DELTA_WEIGHT_RANGE
    else:
        raise ValueError(f"Unknown averaging predictor type: {detector}")


def _weights_from_weight(weight: float) -> np.ndarray:
    return np.array([weight, 1.0 - weight])


if __name__ == "__main__":
    main()
