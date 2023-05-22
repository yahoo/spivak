# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np

from spivak.application.argument_parser import SharedArgs, dir_str_to_path, \
    DATASET_TYPE_CUSTOM_SPOTTING, DATASET_TYPE_CUSTOM_SEGMENTATION, \
    DATASET_TYPE_SOCCERNET_V2_CAMERA_SEGMENTATION, DATASET_TYPE_SOCCERNET_V2, \
    DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION, \
    DATASET_TYPE_SOCCERNET_V2_CHALLENGE, DATASET_TYPE_SOCCERNET, \
    DATASET_TYPE_CUSTOM_SPOTTING_AND_SEGMENTATION, \
    DATASET_TYPE_SOCCERNET_V2_SPOTTING_AND_CAMERA_SEGMENTATION
from spivak.application.dataset_creation import create_label_maps, \
    create_dataset
from spivak.application.model_creation import load_predictor, \
    create_flexible_nms
from spivak.application.validation import save_evaluation_run, EvaluationRun, \
    read_validation_evaluation_run
from spivak.data.dataset import Dataset, VideoDatum, Task, TASK_NAMES
from spivak.data.dataset_splits import SPLIT_KEY_UNLABELED
from spivak.data.label_map import LabelMap
from spivak.data.output_names import OUTPUT_LABEL, OUTPUT_DETECTION_SCORE_NMS, \
    OUTPUT_SEGMENTATION, OUTPUT_DETECTION_SCORE
from spivak.data.soccernet_label_io import \
    write_all_prediction_jsons_for_games, write_all_prediction_jsons
from spivak.data.soccernet_reader import GamePathsReader
from spivak.evaluation.aggregate import EvaluationAggregate
from spivak.evaluation.segmentation_evaluation import \
    create_segmentation_evaluation, SegmentationEvaluation, \
    create_segmentation_evaluation_old
from spivak.evaluation.spotting_evaluation import run_spotting_evaluation, \
    SpottingEvaluation, read_tolerances_config
from spivak.models.dense_predictor import DensePredictor


def translate_dataset_type_to_custom(dataset_type: str) -> str:
    if dataset_type in {
            DATASET_TYPE_CUSTOM_SPOTTING, DATASET_TYPE_SOCCERNET,
            DATASET_TYPE_SOCCERNET_V2,
            DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION,
            DATASET_TYPE_SOCCERNET_V2_CHALLENGE}:
        return DATASET_TYPE_CUSTOM_SPOTTING
    elif dataset_type in {
            DATASET_TYPE_CUSTOM_SEGMENTATION,
            DATASET_TYPE_SOCCERNET_V2_CAMERA_SEGMENTATION}:
        return DATASET_TYPE_CUSTOM_SEGMENTATION
    else:
        return DATASET_TYPE_CUSTOM_SPOTTING_AND_SEGMENTATION


def test(args: SharedArgs) -> None:
    label_maps = create_label_maps(args)
    # Load the dataset
    split = args.test_split
    dataset = create_dataset(args, split, label_maps)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    if args.test_predict:
        _predict_and_save(args, dataset, label_maps, results_dir)
    else:
        # predict_and_save above already runs NMS (optionally) and saves the
        # labels, so we only run the functions below if the command-line
        # options explicitly ask for them.
        if args.test_nms_only:
            _nms_and_save(args, dataset, label_maps, results_dir)
        if args.test_save_labels:
            _save_labels(dataset, results_dir)
    if args.evaluate and not split == SPLIT_KEY_UNLABELED:
        _evaluate(args, dataset, label_maps, results_dir)
    if args.test_save_spotting_jsons:
        if args.dataset_type in {
                DATASET_TYPE_SOCCERNET_V2,
                DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION,
                DATASET_TYPE_SOCCERNET_V2_CHALLENGE,
                DATASET_TYPE_SOCCERNET_V2_SPOTTING_AND_CAMERA_SEGMENTATION}:
            write_all_prediction_jsons_for_games(
                results_dir, dataset.video_data, label_maps[Task.SPOTTING],
                args.frame_rate)
        elif args.dataset_type in {
                DATASET_TYPE_CUSTOM_SPOTTING,
                DATASET_TYPE_CUSTOM_SPOTTING_AND_SEGMENTATION}:
            write_all_prediction_jsons(
                results_dir, dataset.video_data, label_maps[Task.SPOTTING],
                args.frame_rate)


def _predict_and_save(
        args: SharedArgs, dataset: Dataset,
        label_maps: Dict[Task, LabelMap], results_dir: Path) -> None:
    """For all videos of the dataset, run prediction and save the numpy
    arrays with the results"""
    predictor = load_predictor(
        args, label_maps, dataset.input_shape, args.chunk_prediction_border)
    flexible_nms = create_flexible_nms(args, label_maps[Task.SPOTTING])
    logging.info("Going to save individual video results")
    for video_datum in dataset.video_data:
        results_base_path = results_dir / video_datum.relative_path
        results_base_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.predict_video_and_save(
            video_datum, flexible_nms, results_base_path)


def _nms_and_save(
        args: SharedArgs, dataset: Dataset,
        label_maps: Dict[Task, LabelMap], results_dir: Path) -> None:
    flexible_nms = create_flexible_nms(args, label_maps[Task.SPOTTING])
    for video_datum in dataset.video_data:
        results_base_path = results_dir / video_datum.relative_path
        detections_path = _output_path(
            results_base_path, OUTPUT_DETECTION_SCORE)
        video_outputs = {
            OUTPUT_DETECTION_SCORE: np.load(str(detections_path))}
        DensePredictor.save_predictions(
            video_outputs, flexible_nms, results_base_path)


def _evaluate(
        args: SharedArgs, dataset: Dataset,
        label_maps: Dict[Task, LabelMap], results_dir: Path) -> None:
    spotting_evaluation = _spotting_evaluation(
        args, dataset, label_maps.get(Task.SPOTTING), results_dir)
    if args.segmentation_evaluation_old:
        _segmentation_evaluation_old(
            args, dataset, label_maps.get(Task.SEGMENTATION), results_dir)
    segmentation_evaluation = _segmentation_evaluation(
        dataset, label_maps.get(Task.SEGMENTATION), results_dir)
    evaluation_dir = results_dir / "Evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    evaluation = EvaluationAggregate(
        spotting_evaluation, segmentation_evaluation)
    evaluation.save_txt(str(evaluation_dir))
    logging.info("Evaluation result")
    logging.info(evaluation)
    train_args = _read_train_args(args.model)
    evaluation_run = EvaluationRun(evaluation, train_args)
    save_evaluation_run(evaluation_run, str(evaluation_dir))


def _spotting_evaluation(
        args: SharedArgs, dataset: Dataset, label_map: Optional[LabelMap],
        results_dir: Path) -> Optional[SpottingEvaluation]:
    if not label_map:
        return None
    if not _spotting_results_available(results_dir, dataset.video_data):
        return None
    logging.info("Running spotting evaluation")
    # For the all videos of the dataset, run prediction and save the numpy
    # arrays with the results
    detections, labels = _read_spotting_detections_and_labels(
        results_dir, dataset.video_data)
    prune_classes = bool(args.prune_classes)
    create_confusion_data_frame = bool(args.create_confusion)
    config_dir = dir_str_to_path(args.config_dir)
    tolerances_config = read_tolerances_config(config_dir)
    spotting_evaluation = run_spotting_evaluation(
        detections, labels, tolerances_config, args.frame_rate,
        dataset.num_classes_from_task[Task.SPOTTING], prune_classes,
        create_confusion_data_frame=create_confusion_data_frame,
        label_map=label_map)
    logging.info(
        f"Average-mAP: {spotting_evaluation.average_map_dict}")
    return spotting_evaluation


def _save_labels(dataset: Dataset, results_dir: Path) -> None:
    for video_datum in dataset.video_data:
        base_path = results_dir / video_datum.relative_path
        for task in dataset.tasks:
            if video_datum.valid_labels(task):
                labels_path = _labels_path(base_path, task)
                np.save(str(labels_path), video_datum.labels(task))


def _segmentation_evaluation(
        dataset: Dataset, label_map: Optional[LabelMap],
        results_dir: Path) -> Optional[SegmentationEvaluation]:
    if not label_map:
        return None
    if not _segmentation_results_available(results_dir, dataset.video_data):
        return None
    logging.info("Running segmentation evaluation")
    segmentations, labels = _read_segmentations_and_labels(
        results_dir, dataset.video_data)
    segmentation_evaluation = create_segmentation_evaluation(
        segmentations, labels, label_map)
    logging.info(
        f"Segmentation mean IOU: {segmentation_evaluation.mean_iou}")
    return segmentation_evaluation


def _segmentation_evaluation_old(
        args: SharedArgs, dataset: Dataset, label_map: Optional[LabelMap],
        results_dir: Path) -> Optional[SegmentationEvaluation]:
    if not label_map:
        return None
    if not _segmentation_results_available(results_dir, dataset.video_data):
        return None
    logging.info("Running old version (SoccerNet) of segmentation evaluation")
    # Note that labels are not read at this point, since the old evaluation
    # code will directly read the labels using its own particular code flow.
    segmentations = [
        np.load(str(
            _segmentation_path(results_dir / video_datum.relative_path)))
        for video_datum in dataset.video_data
    ]
    list_games = GamePathsReader.read_game_list_v2_camera_segmentation(
        Path(args.splits_dir), args.test_split)
    segmentation_evaluation = create_segmentation_evaluation_old(
        segmentations, label_map, list_games, args.labels_dir, args.frame_rate)
    logging.info(
        f"Segmentation mean IOU (old version): "
        f"{segmentation_evaluation.mean_iou}")
    return segmentation_evaluation


def _read_train_args(model_path: str) -> Optional[SharedArgs]:
    # Read the evaluation_run object from the validation run of the model,
    # then get the args from that.
    validation_evaluation_run = read_validation_evaluation_run(model_path)
    if not validation_evaluation_run:
        return None
    return validation_evaluation_run.train_args


def _segmentation_results_available(
        results_dir: Path, video_data: List[VideoDatum]) -> bool:
    segmentation_path = _segmentation_path(
        results_dir / video_data[0].relative_path)
    return segmentation_path.exists()


def _read_segmentations_and_labels(
        results_dir: Path, video_data: List[VideoDatum]):
    segmentations = []
    labels = []
    for video_datum in video_data:
        base_path = results_dir / video_datum.relative_path
        labels_path = _labels_path(base_path, Task.SEGMENTATION)
        if labels_path.exists():
            labels.append(np.load(str(labels_path)))
            segmentation_path = _segmentation_path(
                results_dir / video_datum.relative_path)
            segmentations.append(np.load(str(segmentation_path)))
    return segmentations, labels


def _spotting_results_available(
        results_dir: Path, video_data: List[VideoDatum]) -> bool:
    spotting_path = _spotting_path(results_dir / video_data[0].relative_path)
    return spotting_path.exists()


def _read_spotting_detections_and_labels(
        results_dir: Path, video_data: List[VideoDatum]):
    detections = []
    labels = []
    for video_datum in video_data:
        base_path = results_dir / video_datum.relative_path
        labels_path = _labels_path(base_path, Task.SPOTTING)
        if labels_path.exists():
            labels.append(np.load(str(labels_path)))
            detections_path = _spotting_path(
                results_dir / video_datum.relative_path)
            detections.append(np.load(str(detections_path)))
    return detections, labels


def _labels_path(base_path: Path, task: Task) -> Path:
    task_name = TASK_NAMES[task]
    return _output_path(base_path, f"{task_name}_{OUTPUT_LABEL}")


def _spotting_path(base_path: Path) -> Path:
    return _output_path(base_path, OUTPUT_DETECTION_SCORE_NMS)


def _segmentation_path(base_path: Path) -> Path:
    return _output_path(base_path, OUTPUT_SEGMENTATION)


def _output_path(base_path: Path, output_name: str) -> Path:
    return base_path.parent / f"{base_path.stem}_{output_name}.npy"
