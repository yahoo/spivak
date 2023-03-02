# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import os
import pickle
from io import StringIO
from typing import Optional, Dict, List

from spivak.evaluation.segmentation_evaluation import SegmentationEvaluation
from spivak.evaluation.spotting_evaluation import SpottingEvaluation
from spivak.evaluation.task_evaluation import TaskEvaluation

EVALUATION_AGGREGATE_TEXT_FILE_NAME = "evaluation_aggregate.txt"
EVALUATION_AGGREGATE_PICKLE_FILE_NAME = "evaluation_aggregate.pkl"


class EvaluationAggregate:

    """Aggregates and summarizes the evaluation results from different tasks."""

    def __init__(
            self, spotting_evaluation: Optional[SpottingEvaluation],
            segmentation_evaluation: Optional[SegmentationEvaluation]) -> None:
        self.spotting_evaluation = spotting_evaluation
        self.segmentation_evaluation = segmentation_evaluation
        if spotting_evaluation:
            tolerances_name = spotting_evaluation.main_tolerances_name
            self.main_metric = spotting_evaluation.average_map_dict[
                tolerances_name]
            self.main_metric_name = (
                f"{SpottingEvaluation.METRIC_AVERAGE_MAP}_{tolerances_name}")
        elif segmentation_evaluation:
            self.main_metric = segmentation_evaluation.mean_iou
            self.main_metric_name = SegmentationEvaluation.METRIC_MEAN_IOU
        else:
            self.main_metric = 0.0
            self.main_metric_name = None
        optional_task_evaluations = [
            spotting_evaluation, segmentation_evaluation]
        self._task_evaluations: List[TaskEvaluation] = [
            task_evaluation for task_evaluation in optional_task_evaluations
            if task_evaluation]

    def scalars_for_logging(self) -> Dict[str, float]:
        return {
            key: value
            for task_evaluation in self._task_evaluations
            for key, value in task_evaluation.scalars_for_logging().items()
        }

    def save_txt(self, save_dir: str) -> None:
        save_path = os.path.join(save_dir, EVALUATION_AGGREGATE_TEXT_FILE_NAME)
        with open(save_path, "w") as txt_file:
            txt_file.write(self.__str__())

    def save_pkl(self, save_dir: str) -> None:
        save_path = os.path.join(
            save_dir, EVALUATION_AGGREGATE_PICKLE_FILE_NAME)
        with open(save_path, "wb") as pkl_file:
            pickle.dump(self, pkl_file)

    def __str__(self):
        with StringIO() as str_io:
            self._write_str(str_io)
            summary = str_io.getvalue()
        return summary

    def _write_str(self, str_io: StringIO) -> None:
        for task_evaluation in self._task_evaluations:
            str_io.write(task_evaluation.summary())
            str_io.write("\n")
