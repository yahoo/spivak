# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import subprocess
from typing import Dict, List, Union, Optional

from spivak.application.argument_parser import DETECTOR_DENSE, \
    DETECTOR_DENSE_DELTA

# Scripts
SCRIPT_TRAIN = "./bin/train.py"
SCRIPT_TEST = "./bin/test.py"
SCRIPT_TRANSFORM = "./bin/transform_features.py"
SCRIPT_CREATE_NORMALIZER = "./bin/create_normalizer.py"
SCRIPT_CREATE_FEATURES_FROM_RESULTS = "./bin/create_features_from_results.py"
SCRIPT_CREATE_AVERAGING_PREDICTOR = "./bin/create_averaging_predictor.py"
# System executables
EXECUTABLE_ZIP = "zip"
# Protocol names, used only for creating directory names
SPOTTING_CHALLENGE = "spotting_challenge"
SPOTTING_CHALLENGE_VALIDATED = "spotting_challenge_validated"
SPOTTING_TEST = "spotting_test"
# Config folders
CONFIG_DIR_CHALLENGE_CONFIDENCE = "soccernet_challenge_confidence"
CONFIG_DIR_CHALLENGE_DELTA = "soccernet_challenge_delta"
CONFIG_DIR_CHALLENGE_DELTA_SOFT_NMS = "soccernet_challenge_delta_soft_nms"
CONFIG_DIR_CONFIDENCE = "soccernet_confidence"
CONFIG_DIR_DELTA = "soccernet_delta"
CONFIG_DIR_DELTA_SOFT_NMS = "soccernet_delta_soft_nms"
# Types for holding command arguments
DictArgs = Dict[str, str]
ListArgs = List[str]
Args = Union[ListArgs, DictArgs]


class Command:

    def __init__(
            self, description: str, executable: str, arguments: Args,
            cwd: Optional[str] = None) -> None:
        self.description = description
        self.executable = executable
        self.arguments = arguments
        self.cwd = cwd

    def run(self) -> subprocess.CompletedProcess:
        command_as_list = self._as_list()
        print(f"Going to run the following command: {self.description}")
        print(" ".join(command_as_list))
        return subprocess.run(command_as_list, cwd=self.cwd)

    def __str__(self) -> str:
        command_as_str = " ".join(self._as_list())
        complete_str = f"Command: {self.description}\n{command_as_str}"
        if self.cwd:
            complete_str = complete_str + f"\nCWD: {self.cwd}"
        return complete_str

    def _as_list(self) -> List[str]:
        if isinstance(self.arguments, dict):
            arguments_list = []
            for key in self.arguments:
                value = self.arguments[key]
                if " " in value:
                    values = value.split(" ")
                else:
                    values = [value]
                new_list = [key] + values
                arguments_list.extend(new_list)
        elif isinstance(self.arguments, list):
            arguments_list = self.arguments
        else:
            raise ValueError(
                f"Command instance arguments is of unexpected type: "
                f"{type(self.arguments)}")
        return [self.executable] + arguments_list


def detector_args(detector: str) -> DictArgs:
    detector_arguments = {"-dc": detector}
    if detector == DETECTOR_DENSE:
        detector_arguments.update({"-cw": "1.0"})
    elif detector == DETECTOR_DENSE_DELTA:
        detector_arguments.update({"-dw": "1.0"})
    return detector_arguments


def create_name(
        parameters: DictArgs, extra_name: str, model_name: str,
        feature_name: str, protocol: str) -> str:
    parameters_string = _parameters_to_string(parameters)
    return (f"{protocol}_{feature_name}_{model_name}_{extra_name}"
            f"_{parameters_string}")


def print_command_list(commands: List[Command]) -> None:
    for command in commands:
        print(command, end="\n\n")


def _parameters_to_string(parameters: DictArgs) -> str:
    parameters_strings = [
         _parameter_to_string(key, parameters[key]) for key in parameters]
    return "_".join(parameters_strings)


def _parameter_to_string(key: str, value: str) -> str:
    clean_key = key.replace("-", "")
    if _has_digit(value):
        separator = ""
    else:
        separator = "_"
    return f"{clean_key}{separator}{value}"


def _has_digit(value: str) -> bool:
    return any(char.isdigit() for char in value)
