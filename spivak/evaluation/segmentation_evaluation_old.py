# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.
#
# This file incorporates work covered by the following copyright and permission
# notice:
#   Copyright (c) 2021 Silvio Giancola
#   Licensed under the terms of the MIT license.
#   You may obtain a copy of the MIT License at https://opensource.org/licenses/MIT

# This file contains pieces of code taken from the following files.
# https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/f28d40e69a27481a82ee45d8d8cfdad1f05b6d4f/Task2-CameraShotSegmentation/CALF-detection/src/train.py
# https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/f28d40e69a27481a82ee45d8d8cfdad1f05b6d4f/Task2-CameraShotSegmentation/CALF-detection/src/metrics_fast.py
# https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/f28d40e69a27481a82ee45d8d8cfdad1f05b6d4f/Task2-CameraShotSegmentation/CALF-detection/src/dataset.py
# https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/f28d40e69a27481a82ee45d8d8cfdad1f05b6d4f/Task2-CameraShotSegmentation/CALF-detection/src/preprocessing.py
# At Yahoo Inc., the code was modified and new code was added. This code can
# now interface with the current evaluation codebase. There are comments
# throughout this file that indicate differences between the code in this file
# and the code in segmentation_evaluation.py, which also runs a similar
# evaluation, but with some small improvements.

import json
import os

import numpy as np
from sklearn.metrics import f1_score

from spivak.data.soccernet_label_io import \
    segmentation_targets_from_change_labels

Camera_Type_DICTIONARY = {
    "Main camera center": 0,
    "Close-up player or field referee": 1,
    "Main camera left": 2,
    "Main camera right": 3,
    "Goal line technology camera": 4,
    "Main behind the goal": 5,
    "Spider camera": 6,
    "Close-up side staff": 7,
    "Close-up corner": 8,
    "Close-up behind the goal": 9,
    "Inside the goal": 10,
    "Public": 11,
    # Difference in implementation here: segmentation_evaluation.py reads in
    # the data using soccernet_constants.py. There, we define
    # CAMERA_DICTIONARY differently, such that the upper-case "Other" also
    # maps to 12. Here, we don't do that, to follow the original
    # implementation of segmentation evaluation.
    # "Other": 12,
    "other": 12,
    "I don't know": 12
}


def run_segmentation_evaluation_old(
        all_predictions, num_classes, list_games, labels_dir, frame_rate):
    intersection_counts_per_class = np.zeros(num_classes, dtype=np.float32)
    union_counts_per_class = np.zeros(num_classes, dtype=np.float32)
    labels_json = "Labels-cameras.json"
    all_targets = []
    all_labels_paths = [
        os.path.join(labels_dir, game, labels_json) for game in list_games]
    for game_index, labels_path in enumerate(all_labels_paths):
        len_half1 = len(all_predictions[2 * game_index])
        len_half2 = len(all_predictions[2 * game_index + 1])
        label_half1, label_half2 = _get_game_labels(
            len_half1, len_half2, labels_path, frame_rate)
        targets1 = segmentation_targets_from_change_labels(label_half1)
        targets2 = segmentation_targets_from_change_labels(label_half2)
        all_targets.extend([targets1, targets2])
        pred_np = all_predictions[2 * game_index].argmax(axis=1)
        target_np = targets1.argmax(axis=1)
        for cl in range(num_classes):
            cur_gt_mask = (target_np == cl)
            cur_pred_mask = (pred_np == cl)
            I = np.sum(np.logical_and(cur_gt_mask, cur_pred_mask),
                       dtype=np.float32)
            U = np.sum(np.logical_or(cur_gt_mask, cur_pred_mask),
                       dtype=np.float32)
            if U > 0:
                intersection_counts_per_class[cl] += I
                union_counts_per_class[cl] += U
        pred_np = all_predictions[2 * game_index + 1].argmax(axis=1)
        target_np = targets2.argmax(axis=1)
        for cl in range(num_classes):
            cur_gt_mask = (target_np == cl)
            cur_pred_mask = (pred_np == cl)
            # Difference in implementation here: the code in
            # segmentation_evaluation.py does not add the
            # mysterious + 1 in the lines below.
            I = np.sum(np.logical_and(cur_gt_mask, cur_pred_mask),
                       dtype=np.float32) + 1
            U = np.sum(np.logical_or(cur_gt_mask, cur_pred_mask),
                       dtype=np.float32) + 1
            if U > 0:
                intersection_counts_per_class[cl] += I
                union_counts_per_class[cl] += U
    per_class_iou = np.divide(
        intersection_counts_per_class, union_counts_per_class)
    mean_iou = np.mean(per_class_iou)
    f1_macro, f1_micro, f1_manual = calculate_f1_scores(
        all_targets, all_predictions, num_classes)
    return f1_macro, f1_micro, f1_manual, mean_iou, per_class_iou


def calculate_f1_scores(all_targets, all_predictions, num_classes):
    total_targets, total_predictions = _append_function(
        all_targets, all_predictions, num_classes)
    # Go from one-hot targets to class integers.
    groundtruth = np.argmax(total_targets, axis=1)
    int_predictions = np.argmax(total_predictions, axis=1)
    f1_macro = f1_score(groundtruth, int_predictions, average='macro')
    f1_micro = f1_score(groundtruth, int_predictions, average='micro')
    f1_manual = _calculate_f1_manual(groundtruth, int_predictions, num_classes)
    return f1_macro, f1_micro, f1_manual


def _calculate_f1_manual(groundtruth, int_detections, num_classes):
    f1_s = np.zeros((num_classes, 1))
    for i in np.arange(num_classes):
        groundtruth_i = groundtruth[np.where(groundtruth == i)[0]] == i
        int_detections_i = int_detections[np.where(groundtruth == i)[0]] == i
        f1_s[i] = f1_score(groundtruth_i, int_detections_i)
    # This ignores the last class: class number 12, which is "Other". I'm not
    # sure why this wasn't also applied when computing the f1 score directly
    # using sklearn. It's also not applied when computing the mean_iou above.
    return np.sum(f1_s[0:12]) / 12


def _append_function(targets, predictions, num_classes):
    total_predictions = np.zeros((1, num_classes))
    total_labels = np.zeros((1, num_classes))
    total_predictions[0, 0] = -1
    for target, prediction in zip(targets, predictions):
        total_predictions = np.append(total_predictions, prediction, axis=0)
        total_labels = np.append(total_labels, target, axis=0)
    return total_labels, total_predictions


def _get_game_labels(len_half1, len_half2, labels_path, frame_rate=2):
    labels = json.load(open(labels_path, "r"))
    # self.list_games = np.load(os.path.join(self.path, split))
    dict_type = Camera_Type_DICTIONARY
    num_classes_segmentation = 13
    label_half1 = np.zeros((len_half1, num_classes_segmentation))
    label_half2 = np.zeros((len_half2, num_classes_segmentation))
    for annotation in labels["annotations"]:
        time = annotation["gameTime"]
        camera_type = annotation["label"]
        half = int(time[0])
        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        frame_rate = int(frame_rate)
        # Difference in implementation here: taking the -1 is not done
        # in segmentation_evaluation.py. I don't think the -1 helps. Going over
        # the feature extraction code, my understanding is that the first frame
        # will be close to second 0, so the -1 should not be needed.
        # frame = frame_rate * (seconds + 60 * minutes)
        frame = frame_rate * (seconds + 60 * minutes) - 1
        if camera_type in dict_type:
            label_type = dict_type[camera_type]
            if half == 1:
                frame = min(frame, len_half1 - 1)
                # Difference in implementation here: my code checks if there
                # is already a label, before adding a new one. Without the
                # check, the same frame can get multiple nearby labels
                # assigned to it, and I prefer to just take the first,
                # which described the camera that came before it.
                # if not np.any(label_half1[frame]):
                label_half1[frame, label_type] = 1
            if half == 2:
                frame = min(frame, len_half2 - 1)
                # Difference in implementation here.
                # if not np.any(label_half2[frame]):
                label_half2[frame, label_type] = 1
    label_half1 = _one_hot_to_all_labels(label_half1)
    label_half2 = _one_hot_to_all_labels(label_half2)
    return label_half1, label_half2


def _one_hot_to_all_labels(onehot):
    nb_frames = onehot.shape[0]
    nb_camera = onehot.shape[1]
    onehot = np.flip(onehot, 0)
    frames_camera = onehot
    camera_type = 0
    camera_length = nb_frames
    count_shot = 0
    for i in range(nb_camera):
        y = onehot[:, i]
        # print('y',y.shape)
        camera_change = np.where(y == 1)[0]
        # print(camera_change.shape)
        if y[camera_change].size > 0:
            if camera_length < camera_change[0]:
                camera_length = camera_change[0]
                camera_type = i
    # print(onehot.shape,range(nb_frames))
    for i in range(nb_frames):
        # print(i)
        x = onehot[i, :]
        loc_events = np.where(x == 1)[0]
        # nb_events = len(loc_events)
        if x[loc_events].size > 0:
            camera_type = loc_events[0]
            count_shot = count_shot + 1
        frames_camera[i, camera_type] = 1
    # print(count_shot,nb_frames)
    return np.flip(frames_camera, 0)
