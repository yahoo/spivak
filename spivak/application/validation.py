# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

# Before importing tensorflow, we need to set the environment variable
# controlling its log level. There is a tensorflow set_verbosity() method
# that is called in compute_validation_result() below, but setting the
# environment variable here before the imports seems to affect different logs
# than setting the variable over there. For some reason, this needs to be set
# in this file (I wasn't able to set it properly outside, before importing
# this file). The possible settings are:
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import os
import pickle
import shutil
import warnings
from typing import List, Optional, Dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import tensorflow as tf

from spivak.application.argument_parser import SharedArgs, dir_str_to_path
from spivak.application.dataset_creation import create_label_maps, \
    create_dataset
from spivak.application.model_creation import create_flexible_nms, \
    load_predictor_from_model_path
from spivak.data.dataset_splits import SPLIT_KEY_VALIDATION
from spivak.evaluation.aggregate import EvaluationAggregate
from spivak.evaluation.spotting_evaluation import run_spotting_evaluation, \
    SpottingEvaluation, read_tolerances_config
from spivak.data.label_map import LabelMap
from spivak.data.output_names import OUTPUT_DETECTION_SCORE, OUTPUT_SEGMENTATION
from spivak.data.dataset import VideoDatum, Dataset, Task
from spivak.evaluation.segmentation_evaluation import \
    create_segmentation_evaluation, SegmentationEvaluation
from spivak.models.dense_predictor import VideoOutputs
from spivak.models.non_maximum_suppression import FlexibleNonMaximumSuppression
from spivak.models.predictor import PredictorInterface

VALIDATION_EVALUATION_DIR = "validation_evaluation"
EVALUATION_RUN_PICKLE_FILE_NAME = "evaluation_run.pkl"
LAST_MODEL_DIR = "last_model"
BEST_MODEL_DIR = "best_model"

MODULE_KERAS_GENERIC_UTILS = "keras.utils.generic_utils"
MODULE_KERAS_ENGINE_TRAINING = "keras.engine.training"
MODULE_KERAS_ENGINE_FUNCTIONAL = "keras.engine.functional"
MODULE_KERAS_LAYER_SERIALIZATION = \
    "keras.saving.saved_model.layer_serialization"


class EvaluationRun:

    def __init__(
            self, evaluation: EvaluationAggregate,
            args: Optional[SharedArgs]) -> None:
        self.evaluation = evaluation
        self.train_args = args


class ValidationResult:

    def __init__(self, evaluation: EvaluationAggregate, epoch: int) -> None:
        self.evaluation = evaluation
        self.epoch = epoch


def compute_validation_result(args, best_metric, epoch):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    logging.getLogger().setLevel(logging.INFO)
    filter_keras_warnings()
    # disable_eager_execution makes inference a lot faster. Note this has to be
    # set inside the function, so that it doesn't affect any module that
    # imports this file.
    tf.compat.v1.disable_eager_execution()
    evaluation = create_evaluation(args, best_metric)
    return ValidationResult(evaluation, epoch)


def filter_keras_warnings() -> None:
    warnings.filterwarnings(
        action="ignore", module=MODULE_KERAS_GENERIC_UTILS)
    warnings.filterwarnings(
        action="ignore", module=MODULE_KERAS_ENGINE_FUNCTIONAL)
    warnings.filterwarnings(
        action="ignore", module=MODULE_KERAS_LAYER_SERIALIZATION)
    warnings.filterwarnings(
        action="ignore", category=UserWarning,
        module=MODULE_KERAS_ENGINE_TRAINING)


def save_evaluation_run(evaluation_run: EvaluationRun, save_dir: str) -> None:
    save_path = os.path.join(save_dir, EVALUATION_RUN_PICKLE_FILE_NAME)
    with open(save_path, "wb") as pkl_file:
        pickle.dump(evaluation_run, pkl_file)


def read_validation_evaluation_run(save_dir: str) -> Optional[EvaluationRun]:
    save_path = os.path.join(
        save_dir, VALIDATION_EVALUATION_DIR, EVALUATION_RUN_PICKLE_FILE_NAME)
    if not os.path.exists(save_path):
        return None
    with open(save_path, "rb") as pkl_file:
        return pickle.load(pkl_file)


def create_evaluation(
        args: SharedArgs, best_metric: float) -> EvaluationAggregate:
    label_maps = create_label_maps(args)
    model_dir = args.model
    last_model_path = os.path.join(model_dir, LAST_MODEL_DIR)
    # So that validation isn't so slow, it has its own chunk prediction border.
    predictor = load_predictor_from_model_path(
        args, last_model_path, label_maps,
        args.validation_chunk_prediction_border)
    evaluation = _run_evaluation(args, predictor, label_maps)
    evaluation_run = EvaluationRun(evaluation, args)
    # Get the path for saving the latest validation result. For consistency
    # with the folder structure in best_model_path, we put the validation
    # results inside the last_model_path folder.
    validation_evaluation_dir = os.path.join(
        last_model_path, VALIDATION_EVALUATION_DIR)
    os.makedirs(validation_evaluation_dir, exist_ok=True)
    save_evaluation_run(evaluation_run, validation_evaluation_dir)
    evaluation.save_txt(validation_evaluation_dir)
    if evaluation.main_metric > best_metric:
        logging.info(
            f" *** Got new best validation metric ("
            f"{evaluation.main_metric_name}) of {evaluation.main_metric}")
        # It's better to save the models here from memory, since by the time
        # validation is completed, a new saved models may have overwritten the
        # previously save one (which was initially loaded above).
        best_model_path = os.path.join(model_dir, BEST_MODEL_DIR)
        predictor.save_model(best_model_path)
        args.save(best_model_path)
        best_validation_evaluation_dir = os.path.join(
            best_model_path, VALIDATION_EVALUATION_DIR)
        if os.path.exists(best_validation_evaluation_dir):
            shutil.rmtree(best_validation_evaluation_dir)
        shutil.copytree(
            validation_evaluation_dir, best_validation_evaluation_dir)
        logging.info(" *** Done saving models and evaluation files.")
    else:
        logging.info(
            f"Validation metric ({evaluation.main_metric_name}): "
            f"{evaluation.main_metric}")
    return evaluation


def create_all_video_outputs(
        video_data: List[VideoDatum], predictor: PredictorInterface
) -> List[VideoOutputs]:
    return [predictor.predict_video(video_datum) for video_datum in video_data]


def create_detections_and_targets(
        video_data: List[VideoDatum], all_video_outputs: List[VideoOutputs],
        flexible_nms: FlexibleNonMaximumSuppression):
    detections = []
    targets = []
    # Loop over all the data and labels
    for video_datum, video_outputs in zip(video_data, all_video_outputs):
        if (not video_datum.valid_labels(Task.SPOTTING) or
                video_datum.labels(Task.SPOTTING) is None):
            logging.warning(
                "Video with no labels inside create_detections_and_targets() "
                "function. Skipping this video.")
            continue
        # Append the results of the predictions to the list
        video_detection_scores = video_outputs[OUTPUT_DETECTION_SCORE]
        # Here, we run NMS once per video, without doing any thresholding.
        # This is in contrast to run_spotting_evaluation_v1, which will run NMS
        # once for each tolerance during evaluation. The approach used here
        # allows us to compute the closest matches only once later below,
        # making the evaluation faster. This style of evaluation is also
        # consistent with how SoccerNet-v2 does their evaluation, and makes
        # it easier to compare across methods.
        video_detection_scores_nms = flexible_nms.maybe_apply(
            video_detection_scores)
        detections.append(video_detection_scores_nms)
        targets.append(video_datum.labels(Task.SPOTTING))
    return detections, targets


def _run_evaluation(
        args: SharedArgs, predictor: PredictorInterface,
        label_maps: Dict[Task, LabelMap]) -> EvaluationAggregate:
    validation_set = create_dataset(args, SPLIT_KEY_VALIDATION, label_maps)
    all_video_outputs = create_all_video_outputs(
        validation_set.video_data, predictor)
    spotting_evaluation = _spotting_evaluation(
        args, validation_set, all_video_outputs, label_maps.get(Task.SPOTTING))
    segmentation_evaluation = _segmentation_evaluation(
        validation_set, all_video_outputs, label_maps.get(Task.SEGMENTATION))
    return EvaluationAggregate(spotting_evaluation, segmentation_evaluation)


def _spotting_evaluation(
        args: SharedArgs, validation_set: Dataset,
        all_video_outputs: List[VideoOutputs], label_map: Optional[LabelMap]
) -> Optional[SpottingEvaluation]:
    if not label_map:
        return None
    existing_output_keys = all_video_outputs[0].keys()
    if OUTPUT_DETECTION_SCORE not in existing_output_keys:
        return None
    # Let's run evaluation for action spotting.
    # Create FlexibleNonMaximumSuppression for validation step.
    flexible_nms = create_flexible_nms(args, label_map)
    detections, targets = create_detections_and_targets(
        validation_set.video_data, all_video_outputs, flexible_nms)
    # Prepare deltas for evaluation
    config_dir = dir_str_to_path(args.config_dir)
    tolerances_config = read_tolerances_config(config_dir)
    return run_spotting_evaluation(
        detections, targets, tolerances_config, args.frame_rate,
        validation_set.num_classes_from_task[Task.SPOTTING],
        bool(args.prune_classes), create_confusion_data_frame=False,
        label_map=label_map)


def _segmentation_evaluation(
        validation_set: Dataset, all_video_outputs: List[VideoOutputs],
        label_map: Optional[LabelMap]) -> Optional[SegmentationEvaluation]:
    if not label_map:
        return None
    existing_output_keys = all_video_outputs[0].keys()
    if OUTPUT_SEGMENTATION not in existing_output_keys:
        return None
    segmentations, labels = _create_segmentations_and_labels(
        validation_set.video_data, all_video_outputs)
    return create_segmentation_evaluation(segmentations, labels, label_map)


def _create_segmentations_and_labels(
        video_data: List[VideoDatum], all_video_outputs: List[VideoOutputs]):
    labels_are_valid = [
        video_datum.valid_labels(Task.SEGMENTATION)
        for video_datum in video_data]
    valid_segmentations = [
        video_outputs[OUTPUT_SEGMENTATION]
        for video_outputs, valid in zip(all_video_outputs, labels_are_valid)
        if valid]
    valid_labels = [
        video_datum.labels(Task.SEGMENTATION)
        for video_datum, valid in zip(video_data, labels_are_valid)
        if valid]
    return valid_segmentations, valid_labels
