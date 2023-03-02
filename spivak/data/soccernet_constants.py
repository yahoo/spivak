# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.
#
# This file incorporates work covered by the following copyright and permission
# notice:
#   Copyright (c) 2021 Silvio Giancola
#   Licensed under the terms of the MIT license.
#   You may obtain a copy of the MIT License at https://opensource.org/licenses/MIT

# This file contains pieces of code taken from the following file. At
# Yahoo Inc., the original code was modified and new code was added.
# https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/20f2f74007c82b68a73c519dff852188df4a8b5a/Task2-CameraShotSegmentation/CALF-segmentation/src/config/classes.py

# Label types
LABEL_FILE_NAME = "Labels.json"  # SoccerNet-v1
LABEL_FILE_NAME_V2 = "Labels-v2.json"
LABEL_FILE_NAME_V2_CAMERAS = "Labels-cameras.json"

# Events as annotated in SoccerNet-v1.
EVENT_DICTIONARY_V1 = {
    "soccer-ball": 0, "soccer-ball-own": 0,
    "r-card": 1, "y-card": 1, "yr-card": 1,
    "substitution-in": 2
}

# Events as annotated in SoccerNet-v2.
EVENT_DICTIONARY_V2 = {
    "Penalty": 0,
    "Kick-off": 1,
    "Goal": 2,
    "Substitution": 3,
    "Offside": 4,
    "Shots on target": 5,
    "Shots off target": 6,
    "Clearance": 7,
    "Ball out of play": 8,
    "Throw-in": 9,
    "Foul": 10,
    "Indirect free-kick": 11,
    "Direct free-kick": 12,
    "Corner": 13,
    "Yellow card": 14,
    "Red card": 15,
    "Yellow->red card": 16
}

CAMERA_DICTIONARY = {
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
    # Note: the original SoccerNet-v2 code only has a lower-case "other" below,
    # whereas if we look at the actual camera labels in the JSON files,
    # we only find the upper-case "Other", but not the lower-case one. Maybe
    # the SoccerNet code was looking to skip "Other" labels, though it doesn't
    # skip the "I don't know" labels.
    "Other": 12,
    "other": 12,
    "I don't know": 12
}
