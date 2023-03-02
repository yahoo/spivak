#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from spivak.application.dataset_creation import read_spotting_label_map
from spivak.data.dataset import Task, INDEX_VALID, INDEX_LABELS, read_num_frames
from spivak.data.dataset_splits import SPLIT_KEY_TEST
from spivak.data.soccernet_label_io import GameSpottingPredictionsReader, \
    SOCCERNET_TYPE_TWO
from spivak.data.soccernet_reader import GameOneHotSpottingLabelReader, \
    GamePathsReader
from spivak.evaluation.aggregate import EvaluationAggregate
from spivak.evaluation.spotting_evaluation import run_spotting_evaluation, \
    SpottingEvaluation, read_tolerances_config

# Command-line arguments.
ARGS_RESULTS_DIR = "results_dir"
ARGS_CONFIG_DIR = "config_dir"
ARGS_LABELS_DIR = "labels_dir"
ARGS_SPLITS_DIR = "splits_dir"
ARGS_OUTPUT_DIR = "output_dir"
ARGS_FEATURES_DIR = "features_dir"
# Other constants.
EVALUATION_FRAME_RATE = 2.0
EVALUATION_FEATURE_NAME = "ResNET_TF2_PCA512"
SOCCERNET_TYPE = SOCCERNET_TYPE_TWO
SLACK_SECONDS = 1.0
REPRODUCE_SOCCERNET_EVALUATION = True


def main() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    args = _get_command_line_arguments()
    config_dir = Path(args[ARGS_CONFIG_DIR])
    spotting_label_map = read_spotting_label_map(config_dir)
    if not spotting_label_map:
        raise ValueError(
            f"Could not read spotting label map from configuration dir "
            f"{config_dir}.")
    logging.info("Going to read detections and labels")
    detections, labels = _read_spotting_detections_and_labels(
        args, spotting_label_map.num_classes())
    tolerances_config = read_tolerances_config(config_dir)
    logging.info("Going to run spotting evaluation")
    spotting_evaluation = run_spotting_evaluation(
        detections, labels, tolerances_config, EVALUATION_FRAME_RATE,
        spotting_label_map.num_classes(), prune_classes=False,
        create_confusion_data_frame=True, label_map=spotting_label_map)
    output_dir = Path(args[ARGS_OUTPUT_DIR])
    _save_and_log_spotting_evaluation(spotting_evaluation, output_dir)


def _read_spotting_detections_and_labels(
        args: Dict, num_classes: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    splits_dir = Path(args[ARGS_SPLITS_DIR])
    results_dir = Path(args[ARGS_RESULTS_DIR])
    labels_dir = Path(args[ARGS_LABELS_DIR])
    features_dir = Path(args[ARGS_FEATURES_DIR])
    game_one_hot_label_reader = GameOneHotSpottingLabelReader(
        SOCCERNET_TYPE, EVALUATION_FRAME_RATE, num_classes)
    game_predictions_reader = GameSpottingPredictionsReader(
        SOCCERNET_TYPE, EVALUATION_FRAME_RATE, num_classes)
    # For evaluation purposes, we currently get the video lengths from the
    # feature files, so we need to know the features_dir, feature_type and
    # frame-rate here.
    game_paths_reader = GamePathsReader(
        SOCCERNET_TYPE, EVALUATION_FEATURE_NAME, features_dir, labels_dir,
        splits_dir)
    valid_game_paths = game_paths_reader.read_valid(SPLIT_KEY_TEST)
    detections = []
    labels = []
    for game_paths in valid_game_paths:
        # Read the number of frames for each video.
        num_video_frames_one = read_num_frames(game_paths.features_one)
        num_video_frames_two = read_num_frames(game_paths.features_two)
        num_label_frames_one = num_video_frames_one
        num_label_frames_two = num_video_frames_two
        if REPRODUCE_SOCCERNET_EVALUATION:
            # In the SoccerNet evaluation code, they don't bother to figure
            # out how many frames are needed, and just use a very large
            # number. This allows labels that are slightly out of bounds to not
            # have to be pushed into bounds, yielding very slightly different
            # results. We add a bit of slack here on the size of the label
            # matrices in order to match their results.
            frame_slack = int(EVALUATION_FRAME_RATE * SLACK_SECONDS)
            num_label_frames_one += frame_slack
            num_label_frames_two += frame_slack
        # Read the labels for the two videos.
        labels_and_valid_one, labels_and_valid_two = \
            game_one_hot_label_reader.read(
                game_paths.labels.get(Task.SPOTTING),
                num_label_frames_one, num_label_frames_two)
        assert labels_and_valid_one[INDEX_VALID]
        assert labels_and_valid_two[INDEX_VALID]
        labels.append(labels_and_valid_one[INDEX_LABELS])
        labels.append(labels_and_valid_two[INDEX_LABELS])
        # Read the detections for the two videos.
        detections_dir = results_dir / game_paths.relative
        detections_path = _spotting_detections_path(detections_dir)
        detections_one, detections_two = game_predictions_reader.read(
            detections_path, num_video_frames_one, num_video_frames_two)
        detections.append(detections_one)
        detections.append(detections_two)
    return detections, labels


def _spotting_detections_path(detections_dir: Path) -> Path:
    glob_result = list(detections_dir.glob("*.json"))
    assert len(glob_result) == 1
    return glob_result[0]


def _save_and_log_spotting_evaluation(
        spotting_evaluation: SpottingEvaluation, output_dir: Path):
    logging.info(f"Average-mAP V2: {spotting_evaluation.average_map_dict}")
    evaluation_dir = output_dir / "Evaluation"
    evaluation_dir.mkdir(exist_ok=True, parents=True)
    evaluation = EvaluationAggregate(
        spotting_evaluation, segmentation_evaluation=None)
    logging.info("Evaluation result")
    logging.info(evaluation)
    evaluation.save_txt(str(evaluation_dir))
    evaluation.save_pkl(str(evaluation_dir))


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + ARGS_RESULTS_DIR, required=True,
        help='Input directory containing JSON results')
    parser.add_argument(
        "--" + ARGS_FEATURES_DIR, required=True,
        help='Input directory containing features')
    parser.add_argument(
        "--" + ARGS_LABELS_DIR, required=True,
        help='Input directory containing labels')
    parser.add_argument(
        "--" + ARGS_SPLITS_DIR, required=True,
        help="Directory containing file(s) with splits definitions.")
    parser.add_argument(
        "--" + ARGS_OUTPUT_DIR, required=True,
        help="Output directory, for saving the evaluation result files.")
    parser.add_argument(
        "--" + ARGS_CONFIG_DIR, required=True,
        help="Directory with configuration files")
    args_dict = vars(parser.parse_args())
    return args_dict


if __name__ == "__main__":
    main()
