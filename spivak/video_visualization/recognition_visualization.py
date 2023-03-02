# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import math
from typing import List, Optional
from typing import Tuple

import cv2
import numpy as np

from spivak.data.label_map import LabelMap

LABEL_SEPARATOR = ";"
LABEL_TEXT_RELATIVE_GAP_X = 30
BOTTOM_LABEL_TEXT_RELATIVE_GAP_Y = 20
TARGET_FONT_SCALE = 1.0
SMALL_TARGET_FONT_SCALE = 0.6
BOLD_FONT_RELATIVE_THICKNESS = 2.0
# Avoid changing the font, as it will change other constants below
# (unfortunately, the font size changes with the font choice).
FONT = cv2.FONT_HERSHEY_SIMPLEX
# Don't make the font too small, so the number of pixels in the text isn't
# so small that it becomes unreadable.
FONT_SCALE_MIN_ABSOLUTE = 0.3
# Don't make the font too small, relative to the image size.
FONT_SCALE_MIN_RELATIVE = 5e-4
# Don't make the font too large, relative to the image size.
FONT_SCALE_MAX_RELATIVE = 1e-3
CV_COLOR_TEXT = (255, 255, 255)

OpenCVTextSizes = List[Tuple[Tuple[int, int], int]]


class Label:

    def __init__(self, text: str, bold: bool) -> None:
        self.text = text
        self.bold = bold


class FrameRecognizedActionsView:

    def __init__(self, time_in_seconds: float, scores: np.ndarray,
                 label_map: LabelMap) -> None:
        self.time_in_seconds = time_in_seconds
        self.scores = scores
        self.label_map = label_map


def get_label(
        score: float, recognition_threshold: float, class_name: str) -> Label:
    return Label(create_label_text(score, class_name),
                 score > recognition_threshold)


def get_multi_labels(
        frames_actions_view: FrameRecognizedActionsView,
        recognition_threshold: float) -> List[Label]:
    return [get_label(
        score, recognition_threshold,
        frames_actions_view.label_map.int_to_label[index])
        for index, score in enumerate(frames_actions_view.scores)]


def get_standard_font_thickness(font_scale: float) -> int:
    return max(1, int(font_scale))


def get_bold_font_thickness(
        standard_font_thickness: int, font_scale: float) -> int:
    bold_font_thickness = int(font_scale * BOLD_FONT_RELATIVE_THICKNESS)
    if bold_font_thickness == standard_font_thickness:
        return standard_font_thickness + 1
    return bold_font_thickness


def get_text_size(
        label: Label, font_scale: float, standard_font_thickness: int,
        bold_font_thickness: int) -> OpenCVTextSizes:
    if label.bold:
        effective_thickness = bold_font_thickness
    else:
        effective_thickness = standard_font_thickness
    return cv2.getTextSize(label.text, FONT, font_scale, effective_thickness)


def cv2_draw_labels(
        np_image: np.ndarray, frame_time: float,
        labels: Optional[List[Label]]) -> None:
    if not labels:
        return
    image_height, image_width, _ = np_image.shape
    font_scale = compute_font_scale(image_height, image_width)
    standard_font_thickness = get_standard_font_thickness(font_scale)
    bold_font_thickness = get_bold_font_thickness(
        standard_font_thickness, font_scale)
    text_sizes = [
        get_text_size(label, font_scale, standard_font_thickness,
                      bold_font_thickness) for label in labels]
    baselines = np.array([t[1] for t in text_sizes])
    max_baseline = max(baselines)
    location_y = math.ceil(
        image_height - max_baseline - font_scale *
        BOTTOM_LABEL_TEXT_RELATIVE_GAP_Y)
    location_xs = compute_labels_xs(font_scale, text_sizes)
    for i, (label, location_x) in enumerate(zip(labels, location_xs)):
        if label.bold:
            font_thickness = bold_font_thickness
        else:
            font_thickness = standard_font_thickness
        text = label.text
        # Add the separator to all labels except the last one.
        if i != len(labels) - 1:
            text += LABEL_SEPARATOR
        cv2.putText(
            np_image, text, (location_x, location_y), FONT, font_scale,
            CV_COLOR_TEXT, font_thickness)


def compute_labels_xs(
        font_scale: float, text_sizes: List[OpenCVTextSizes]) -> List[int]:
    label_widths = np.array([t[0][0] for t in text_sizes])
    relative_shifts = np.insert(label_widths[:-1], 0, 0)
    relative_shifts_with_gaps = (
            relative_shifts + font_scale * LABEL_TEXT_RELATIVE_GAP_X)
    label_shifts = np.cumsum(relative_shifts_with_gaps)
    return [int(np.rint(shift)) for shift in label_shifts]


def create_label_text(score: float, class_name: str) -> str:
    return "[{:.2f}]: {:s}".format(score, class_name)


def compute_font_scale(image_height: int, image_width: int) -> float:
    if image_height > image_width:
        # We have a tall video, so let's make the font smaller so that more
        # text will fit in the bottom.
        target_font_scale = SMALL_TARGET_FONT_SCALE
    else:
        target_font_scale = TARGET_FONT_SCALE
    return _limit_font_scale(target_font_scale, image_height)


def _limit_font_scale(target_font_scale: float, image_height: int) -> float:
    min_font_scale = max(
        FONT_SCALE_MIN_RELATIVE * image_height, FONT_SCALE_MIN_ABSOLUTE)
    max_font_scale = FONT_SCALE_MAX_RELATIVE * image_height
    return min(max(min_font_scale, target_font_scale), max_font_scale)
