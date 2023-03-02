# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from io import StringIO
from typing import Dict, List

import numpy as np

from spivak.data.label_map import LabelMap
from spivak.data.soccernet_label_io import \
    segmentation_targets_from_change_labels
from spivak.evaluation.segmentation_evaluation_old import \
    run_segmentation_evaluation_old, calculate_f1_scores
from spivak.evaluation.task_evaluation import TaskEvaluation


class SegmentationEvaluation(TaskEvaluation):

    METRIC_MEAN_IOU = "mean_iou"
    METRIC_IOU = "iou"

    def __init__(
            self, mean_iou: float, per_class_iou: np.ndarray,
            label_map: LabelMap) -> None:
        self.mean_iou = mean_iou
        self.per_class_iou = per_class_iou
        self.label_map = label_map

    def scalars_for_logging(self) -> Dict[str, float]:
        scalars = {
            f"{SegmentationEvaluation.METRIC_IOU}_"
            f"{self.label_map.int_to_label[c]}": class_iou
            for c, class_iou in enumerate(self.per_class_iou)
        }
        scalars[SegmentationEvaluation.METRIC_MEAN_IOU] = self.mean_iou
        return scalars

    def summary(self) -> str:
        with StringIO() as str_io:
            self._write_summary(str_io)
            summary = str_io.getvalue()
        return summary

    def _write_summary(self, str_io: StringIO) -> None:
        str_io.write("Segmentation evaluation:\n")
        str_io.write(
            f"{SegmentationEvaluation.METRIC_MEAN_IOU}: {self.mean_iou}\n")
        str_io.write("\nIoU per class:\n")
        for class_index, class_iou in enumerate(self.per_class_iou):
            class_name = self.label_map.int_to_label[class_index]
            str_io.write(f"{class_name}: {class_iou}\n")


def create_segmentation_evaluation(
        all_segmentations: List[np.ndarray], all_labels: List[np.ndarray],
        label_map: LabelMap) -> SegmentationEvaluation:
    # f1_manual ignores the last class, not sure if we want to do that,
    # so can maybe just ignore it.
    f1_macro, f1_micro, f1_manual, mean_iou, per_class_iou = \
        _run_segmentation_evaluation(
            all_segmentations, all_labels, label_map.num_classes())
    return SegmentationEvaluation(mean_iou, per_class_iou, label_map)


def create_segmentation_evaluation_old(
        all_segmentations: List[np.ndarray], label_map: LabelMap,
        list_games, labels_dir, frame_rate) -> SegmentationEvaluation:
    # Replicates the SoccerNet code evaluation as of August 23, 2022. That code
    # had some small problems, which are noted inside individual comments in
    # segmentation_evaluation_old.py.
    f1_macro, f1_micro, f1_manual, mean_iou, per_class_iou = \
        run_segmentation_evaluation_old(
            all_segmentations, label_map.num_classes(), list_games, labels_dir,
            frame_rate)
    return SegmentationEvaluation(mean_iou, per_class_iou, label_map)


def _run_segmentation_evaluation(all_predictions, all_labels, num_classes):
    intersection_counts_per_class = np.zeros(num_classes, dtype=np.float32)
    union_counts_per_class = np.zeros(num_classes, dtype=np.float32)
    all_targets = [
        segmentation_targets_from_change_labels(labels)
        for labels in all_labels]
    for (video_predictions, video_targets) in zip(all_predictions, all_targets):
        # Convert from one-hot to integers.
        video_predictions_integers = video_predictions.argmax(axis=1)
        video_targets_integers = video_targets.argmax(axis=1)
        for class_index in range(num_classes):
            target_mask = (video_targets_integers == class_index)
            prediction_mask = (video_predictions_integers == class_index)
            intersection_count = np.sum(
                np.logical_and(target_mask, prediction_mask), dtype=np.float32)
            union_count = np.sum(
                np.logical_or(target_mask, prediction_mask), dtype=np.float32)
            intersection_counts_per_class[class_index] += intersection_count
            union_counts_per_class[class_index] += union_count
    per_class_iou = np.divide(
        intersection_counts_per_class, union_counts_per_class)
    mean_iou = float(np.mean(per_class_iou))
    f1_macro, f1_micro, f1_manual = calculate_f1_scores(
        all_targets, all_predictions, num_classes)
    return f1_macro, f1_micro, f1_manual, mean_iou, per_class_iou
