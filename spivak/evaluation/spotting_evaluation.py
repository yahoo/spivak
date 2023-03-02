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
# https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/20f2f74007c82b68a73c519dff852188df4a8b5a/Task1-ActionSpotting/CALF/src/metrics_visibility_fast.py
# At Yahoo Inc., the code was modified and new code was added. The result of
# the current evaluation code should match that of the original code from
# SoccerNet-v2, while being faster to run. It can optionally also save a series
# of intermediate evaluation metrics that can later be used to create
# visualizations.

import json
import pickle
from collections import Counter
from io import StringIO
from itertools import takewhile
from pathlib import Path
from typing import Optional, List, Any, Dict

import numpy as np
from pandas import DataFrame

from spivak.data.label_map import LabelMap
from spivak.evaluation.task_evaluation import TaskEvaluation

np.seterr(divide='ignore', invalid='ignore')

EVALUATION_TOLERANCES_JSON = "evaluation_tolerances.json"
TOLERANCES_JSON_KEY_MAIN = "main"
TOLERANCES_JSON_KEY_EXTRA = "extra"
TOLERANCES_JSON_KEY_SETS = "sets"
CLASS_PRUNE_COUNT_THRESHOLD = 10


class TolerancesConfig:

    def __init__(
            self, tolerance_targets_in_seconds_dict: Dict[str, np.ndarray],
            main_tolerances_name: str,
            extra_tolerance_targets_in_seconds: np.ndarray) -> None:
        self.tolerance_targets_in_seconds_dict = \
            tolerance_targets_in_seconds_dict
        self.main_tolerances_name = main_tolerances_name
        self.extra_tolerance_targets_in_seconds = \
            extra_tolerance_targets_in_seconds


class SpottingEvaluation(TaskEvaluation):
    TOLERANCE = "Tolerance"
    TOLERANCE_INT = "Tolerance integer"
    CLASS_INDEX = "Class index"
    CLASS_NAME = "Class"
    VIDEO = "Video"
    THRESHOLD = "Threshold"
    CONDITION_POSITIVE = "Condition positive"
    PREDICTED_POSITIVE = "Predicted positive"
    TRUE_POSITIVE = "True positive"
    AVERAGE_PRECISION = "Average precision"
    PRECISION = "Precision"
    RECALL = "Recall"

    METRIC_AVERAGE_MAP = "average_map"
    METRIC_AVERAGE_AP = "average_ap"
    METRIC_MAP_TOLERANCE = "map_tolerance"

    def __init__(
            self, average_map_dict: Dict[str, float],
            class_average_ap_dict: Dict[str, List[float]],
            tolerance_map: Dict[float, float], ap_data_frame: DataFrame,
            pr_data_frame: DataFrame, confusion_data_frame: Optional[DataFrame],
            selected_classes: List[bool],
            tolerances_in_seconds_dict: Dict[str, np.ndarray],
            main_tolerances_name: str, label_map: LabelMap) -> None:
        self.average_map_dict = average_map_dict
        self.class_average_ap_dict = class_average_ap_dict
        self.tolerance_map = tolerance_map
        self.ap_data_frame = ap_data_frame
        self.pr_data_frame = pr_data_frame
        self.confusion_data_frame = confusion_data_frame
        self.selected_classes = selected_classes
        self.tolerances_in_seconds_dict = tolerances_in_seconds_dict
        self.main_tolerances_name = main_tolerances_name
        self.label_map = label_map

    def scalars_for_logging(self) -> Dict[str, float]:
        scalars_to_write = {
            f"{SpottingEvaluation.METRIC_AVERAGE_AP}_{tolerances_name}_"
            f"{self.label_map.int_to_label[c]}": average_ap
            for tolerances_name, class_average_ap in
            self.class_average_ap_dict.items()
            for c, average_ap in enumerate(class_average_ap)
        }
        scalars_to_write.update({
            f"{SpottingEvaluation.METRIC_MAP_TOLERANCE}_{tolerance}": mean_ap
            for tolerance, mean_ap in self.tolerance_map.items()
        })
        scalars_to_write.update({
            f"{SpottingEvaluation.METRIC_AVERAGE_MAP}_{tolerances_name}":
                average_map
            for tolerances_name, average_map in self.average_map_dict.items()
        })
        return scalars_to_write

    def save_txt(self, save_path: str) -> None:
        with open(save_path, "w") as txt_file:
            txt_file.write(self.summary())

    def save_pkl(self, save_path: str) -> None:
        with open(save_path, "wb") as pkl_file:
            pickle.dump(self, pkl_file)

    def summary(self) -> str:
        with StringIO() as str_io:
            self._write_summary(str_io)
            summary = str_io.getvalue()
        return summary

    def _write_summary(self, str_io: StringIO) -> None:
        str_io.write("Spotting evaluation:\n")
        str_io.write("\nClasses used when computing mAP:\n")
        for class_index, class_is_selected in enumerate(
                self.selected_classes):
            if class_is_selected:
                class_name = self.label_map.int_to_label[class_index]
                str_io.write(f"{class_name}  ")
            str_io.write("\n")
        str_io.write("\nTolerances used when computing averages:\n")
        for tolerances_name, tolerances_in_seconds in \
                self.tolerances_in_seconds_dict.items():
            str_io.write(f"{tolerances_name}: {tolerances_in_seconds}\n")
        str_io.write(f"\nAverage-mAP: {self.average_map_dict}\n")
        for tolerances_name, class_average_ap in \
                self.class_average_ap_dict.items():
            str_io.write(
                f"\n{tolerances_name} tolerances average AP per class:\n")
            for class_index, average_ap in enumerate(class_average_ap):
                class_name = self.label_map.int_to_label[class_index]
                str_io.write(f"{class_name}: {average_ap}\n")
        str_io.write("\nAverage mAP per matching tolerance:\n")
        tolerances = sorted(self.tolerance_map.keys())
        for tolerance in tolerances:
            str_io.write(
                f"{tolerance}: {self.tolerance_map[tolerance]}\n")


def read_tolerances_config(config_dir: Path) -> TolerancesConfig:
    tolerances_config_data = _read_tolerances_json(config_dir)
    tolerance_targets_in_seconds_map = {
        tolerances_name: np.array(tolerances, dtype=float)
        for tolerances_name, tolerances in tolerances_config_data[
            TOLERANCES_JSON_KEY_SETS].items()
    }
    main_tolerances_name = tolerances_config_data[TOLERANCES_JSON_KEY_MAIN]
    extra_tolerance_targets_in_seconds = np.array(
        tolerances_config_data[TOLERANCES_JSON_KEY_EXTRA], dtype=float)
    return TolerancesConfig(
        tolerance_targets_in_seconds_map, main_tolerances_name,
        extra_tolerance_targets_in_seconds)


def run_spotting_evaluation(
        detections: List[np.ndarray], targets: List[np.ndarray],
        tolerances_config: TolerancesConfig, frame_rate: float,
        num_classes: int, prune_classes: bool,
        create_confusion_data_frame: bool,
        label_map: Optional[LabelMap] = None) -> SpottingEvaluation:
    if create_confusion_data_frame:
        confusion_rows = []
    else:
        confusion_rows = None
    all_tolerances = _collect_all_tolerances(tolerances_config, frame_rate)
    all_ap, pr_rows = _compute_all_ap(
        targets, detections, all_tolerances, confusion_rows)
    # Select relevant classes and average over them.
    if prune_classes:
        selected_classes = _select_classes(targets, num_classes)
    else:
        selected_classes = [True] * num_classes
    average_map_dict = {}
    class_average_ap_dict = {}
    tolerances_in_seconds_dict = {}
    for tolerances_name, tolerance_targets_in_seconds in \
            tolerances_config.tolerance_targets_in_seconds_dict.items():
        tolerances = _tolerances_from_targets_in_seconds(
            tolerance_targets_in_seconds, frame_rate)
        tolerances_selection = [
            np.where(all_tolerances == tolerance)[0][0]
            for tolerance in tolerances]
        ap = all_ap[tolerances_selection]
        mean_ap = np.mean(ap[:, selected_classes], axis=1)
        # Fancy average over tolerances.
        average_map_dict[tolerances_name] = _average_over_tolerance(
            mean_ap, tolerances)
        class_average_ap_dict[tolerances_name] = [
            _average_over_tolerance(ap[:, class_index], tolerances)
            for class_index in range(num_classes)]
        tolerances_in_seconds_dict[tolerances_name] = tolerances / frame_rate
    # For reporting, get the actual tolerances used, but in seconds
    all_tolerances_in_seconds = all_tolerances / frame_rate
    tolerance_map = {
        tolerance_in_seconds: np.mean(
            all_ap[tolerance_index, selected_classes])
        for tolerance_index, tolerance_in_seconds in enumerate(
            all_tolerances_in_seconds)}
    if create_confusion_data_frame:
        confusion_data_frame = _create_confusion_data_frame(
            confusion_rows, frame_rate)
    else:
        confusion_data_frame = None
    ap_data_frame = _create_ap_data_frame(
        all_ap, all_tolerances_in_seconds, label_map)
    pr_data_frame = _create_pr_data_frame(pr_rows, frame_rate)
    return SpottingEvaluation(
        average_map_dict, class_average_ap_dict, tolerance_map, ap_data_frame,
        pr_data_frame, confusion_data_frame, selected_classes,
        tolerances_in_seconds_dict, tolerances_config.main_tolerances_name,
        label_map)


def _collect_all_tolerances(
        tolerances_config: TolerancesConfig, frame_rate: float) -> np.ndarray:
    tolerance_targets_in_seconds_arrays = list(
        tolerances_config.tolerance_targets_in_seconds_dict.values()) + [
        tolerances_config.extra_tolerance_targets_in_seconds]
    collected_tolerance_targets_in_seconds = np.concatenate(
        tolerance_targets_in_seconds_arrays)
    collected_tolerances = _tolerances_from_targets_in_seconds(
        collected_tolerance_targets_in_seconds, frame_rate)
    # This also sorts the tolerances.
    return np.unique(collected_tolerances)


def _tolerances_from_targets_in_seconds(
        tolerance_targets_in_seconds: np.ndarray,
        frame_rate: float) -> np.ndarray:
    tolerance_targets = frame_rate * tolerance_targets_in_seconds
    # Convert the tolerances to integers for speed in later calculations. We
    # decide to round down, which should not affect the matching process,
    # since all the indexes being matched are already integers.
    return tolerance_targets.astype(int)


def _create_confusion_data_frame(
        confusion_rows: List[Dict], frame_rate: float) -> DataFrame:
    data_frame = DataFrame(confusion_rows)
    return _add_float_tolerance(data_frame, frame_rate)


def _compute_all_ap(
        targets: List[np.ndarray], detections: List[np.ndarray],
        tolerances: np.ndarray, confusion_rows: Optional[List[Dict[str, Any]]]):
    pr_rows = []
    n_classes = targets[0].shape[1]
    n_tolerances = len(tolerances)
    all_ap = np.zeros((n_tolerances, n_classes))
    for tolerance_index, tolerance in enumerate(tolerances):
        precision, recall, tolerance_pr_rows = _compute_precision_recall_curve(
            targets, detections, tolerance, confusion_rows)
        pr_rows.extend(tolerance_pr_rows)
        all_ap[tolerance_index, :] = _compute_tolerance_ap(precision, recall)
    return all_ap, pr_rows


def _compute_precision_recall_curve(
        targets, detections, tolerance, confusion_rows):
    thresholds = np.linspace(0, 1, 200)
    precision, recall = _precision_recall_core(
        targets, detections, tolerance, thresholds, confusion_rows)
    pr_rows = _create_pr_rows(tolerance, precision, recall, thresholds)
    return precision, recall, pr_rows


def _precision_recall_core(
        targets, detections, tolerance, thresholds, confusion_rows):
    num_videos = len(targets)
    precision = list()
    recall = list()
    num_classes = targets[0].shape[-1]
    # Pre-compute the prediction scores and their correspondence for each class
    for c in np.arange(num_classes):
        precision.append(list())
        recall.append(list())
        total_detections = np.zeros((1, 3))
        total_detections[0, 0] = -1
        all_video_gt_labels = np.zeros(num_videos)
        # Get the confidence scores and their corresponding true positive or
        # false positive characteristics for each video.
        for v, (target, detection) in enumerate(zip(targets, detections)):
            tmp_detections, video_gt_labels = _compute_class_scores(
                target[:, c], detection[:, c], tolerance, v)
            total_detections = np.append(
                total_detections, tmp_detections, axis=0)
            all_video_gt_labels[v] = video_gt_labels
        total_gt_labels = np.sum(all_video_gt_labels)
        # Get the precision and recall for each confidence threshold
        for threshold in thresholds:
            pred_indexes = np.where(total_detections[:, 0] >= threshold)[0]
            thresholded_detections = total_detections[pred_indexes]
            if confusion_rows is not None:
                predicted_positive_video_indexes = thresholded_detections[:, 2]
                predicted_positive_per_video = Counter(
                    predicted_positive_video_indexes)
                true_positive_video_indexes = predicted_positive_video_indexes[
                    thresholded_detections[:, 1] == 1]
                true_positive_per_video = Counter(true_positive_video_indexes)
                for v in range(num_videos):
                    confusion_rows.append({
                        SpottingEvaluation.TOLERANCE_INT: tolerance,
                        SpottingEvaluation.CLASS_INDEX: c,
                        SpottingEvaluation.VIDEO: v,
                        SpottingEvaluation.CONDITION_POSITIVE:
                            all_video_gt_labels[v],
                        SpottingEvaluation.PREDICTED_POSITIVE:
                            predicted_positive_per_video[v],
                        SpottingEvaluation.THRESHOLD: threshold,
                        SpottingEvaluation.TRUE_POSITIVE:
                            true_positive_per_video[v]
                    })
            true_positives = np.sum(thresholded_detections[:, 1])
            p = np.nan_to_num(true_positives / len(pred_indexes))
            r = np.nan_to_num(true_positives / total_gt_labels)
            precision[-1].append(p)
            recall[-1].append(r)
    precision = np.array(precision).transpose()
    recall = np.array(recall).transpose()
    return precision, recall


def _compute_class_scores(target, detection, tolerance, video_index):
    # Retrieving the important variables
    gt_indexes = np.where(target != 0)[0]
    pred_indexes = np.where(detection >= 0)[0]
    pred_scores = detection[pred_indexes]
    # Array to save the results, each is [pred_score, {1 or 0}, video_index]
    scores_info = np.zeros((len(pred_indexes), 3))
    scores_info[:, 0] = np.copy(pred_scores)
    # Tracks which video the score came from
    scores_info[:, 2] = video_index
    remove_indexes = list()
    for gt_index in gt_indexes:
        # Find the best prediction for the ground-truth instance, if there is
        # one available that is within the tolerance range and not yet
        # accounted for (not in remove_indexes). Note that each ground truth
        # will result in at most one prediction match, resulting in at most one
        # true positive per ground-truth. Similarly, each prediction will
        # correspond to at most one ground-truth, since once it is matched
        # to a ground-truth, it will be accounted for in remove_indexes.
        max_index, max_index_pos = _find_best_match(
            gt_index, pred_indexes, pred_scores, tolerance, remove_indexes)
        if max_index is not None:
            # We found a matching prediction
            scores_info[max_index_pos, 1] = 1
            remove_indexes.append(max_index)
    return scores_info, len(gt_indexes)


def _find_best_match(
        gt_index, pred_indexes, pred_scores, tolerance, remove_indexes):
    half_tolerance = tolerance // 2
    # The interval over the tolerance is inclusive, so we will accept
    # first_index and last_index as defined below, as well as anything in
    # between.
    first_index = gt_index - half_tolerance
    last_index = gt_index + half_tolerance
    # When using "left" below, we get pred_indexes[first_pos] greater or
    # equal to first_index, so we want to accept first_pos.
    first_pos = np.searchsorted(pred_indexes, first_index, "left")
    # Going from first_pos until pred_indexes is larger than last_index.
    nearby_positions = takewhile(
        lambda pos: pred_indexes[pos] <= last_index,
        range(first_pos, len(pred_indexes)))
    # For each nearby position that is not to be removed, generate a triplet
    # with its position, index and score.
    triples_generator = (
        (pred_scores[pos], pred_indexes[pos], pos)
        for pos in nearby_positions
        if pred_indexes[pos] not in remove_indexes)
    # Get the nearby triplet with the max score. max will return one triplet,
    # which corresponds to a single prediction.
    max_score, max_index, max_index_pos = max(
        triples_generator, key=lambda score_index_pos: score_index_pos[0],
        default=(None, None, None))
    return max_index, max_index_pos


def _select_classes(targets: List[np.ndarray], n_classes: int) -> List[bool]:
    class_counts = [
        sum(np.sum(video_targets[:, class_index]) for video_targets in targets)
        for class_index in range(n_classes)]
    return [count > CLASS_PRUNE_COUNT_THRESHOLD for count in class_counts]


def _average_over_tolerance(mean_ap, tolerances):
    # Compute the average mAP
    integral = 0.0
    total_step_sizes = 0.0
    for i in np.arange(len(mean_ap) - 1):
        step_size = tolerances[i + 1] - tolerances[i]
        integral += step_size * (mean_ap[i] + mean_ap[i + 1]) / 2
        total_step_sizes += step_size
    average_map = integral / total_step_sizes
    return average_map


def _compute_tolerance_ap(precision, recall):
    _sort_precision_recall(precision, recall)

    # Array for storing the ap per class
    ap = np.array([0.0] * precision.shape[-1])

    # Loop for all classes
    for i in np.arange(precision.shape[-1]):

        # 11 point interpolation
        for j in np.arange(11) / 10:

            index_recall = np.where(recall[:, i] >= j)[0]

            possible_value_precision = precision[index_recall, i]
            max_value_precision = 0

            if possible_value_precision.shape[0] != 0:
                max_value_precision = np.max(possible_value_precision)

            ap[i] += max_value_precision

    tolerance_ap = ap / 11

    return tolerance_ap


def _create_ap_data_frame(ap, tolerances_in_seconds, label_map: LabelMap):
    ap_data = [
        {
            SpottingEvaluation.TOLERANCE: tolerance_in_seconds,
            SpottingEvaluation.CLASS_INDEX: class_index,
            SpottingEvaluation.AVERAGE_PRECISION: ap[t, class_index]
        }
        for t, tolerance_in_seconds in enumerate(tolerances_in_seconds)
        for class_index in range(label_map.num_classes())
    ]
    return DataFrame(ap_data)


def _create_pr_rows(tolerance, precision, recall, thresholds):
    num_classes = precision.shape[1]
    return [
        {
            SpottingEvaluation.TOLERANCE_INT: tolerance,
            SpottingEvaluation.PRECISION: precision[t, c],
            SpottingEvaluation.RECALL: recall[t, c],
            SpottingEvaluation.THRESHOLD: threshold,
            SpottingEvaluation.CLASS_INDEX: c
        }
        for t, threshold in enumerate(thresholds)
        for c in range(num_classes)
    ]


def _create_pr_data_frame(pr_rows: List[Dict], frame_rate: float) -> DataFrame:
    data_frame = DataFrame(pr_rows)
    return _add_float_tolerance(data_frame, frame_rate)


def _add_float_tolerance(
        data_frame: DataFrame, frame_rate: float) -> DataFrame:
    data_frame[SpottingEvaluation.TOLERANCE] = (
            data_frame[SpottingEvaluation.TOLERANCE_INT] / frame_rate)
    return data_frame


def _sort_precision_recall(precision, recall):
    # Sort the points based on the recall, class per class
    for i in np.arange(precision.shape[1]):
        index_sort = np.argsort(recall[:, i])
        precision[:, i] = precision[index_sort, i]
        recall[:, i] = recall[index_sort, i]


def _read_tolerances_json(config_dir: Path) -> Dict:
    json_path = config_dir / EVALUATION_TOLERANCES_JSON
    with json_path.open("r") as json_file:
        return json.load(json_file)
