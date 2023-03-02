# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np

from spivak.data.dataset import VideoDatum
from spivak.models.non_maximum_suppression import FlexibleNonMaximumSuppression

VideoOutputs = Dict[str, np.ndarray]


class PredictorInterface(metaclass=ABCMeta):

    @abstractmethod
    def predict_video(self, video_datum: VideoDatum) -> VideoOutputs:
        pass

    @abstractmethod
    def predict_video_and_save(
            self, video_datum: VideoDatum, nms: FlexibleNonMaximumSuppression,
            base_path: Path) -> None:
        pass

    @abstractmethod
    def load_weights(self, weights_path: str) -> None:
        pass

    @abstractmethod
    def save_model(self, model_path: str) -> None:
        pass
