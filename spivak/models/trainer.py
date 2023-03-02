# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from abc import abstractmethod
from typing import List, Optional

from tensorflow.keras import Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Optimizer

from spivak.models.assembly.head import TrainerHeadInterface
from spivak.models.tf_dataset import TFDataset


class TrainerInterface:

    @abstractmethod
    def compile(self, optimizer: Optimizer) -> None:
        pass

    @abstractmethod
    def fit(self, initial_epoch: int, epochs: int, callbacks,
            validation_freq) -> History:
        pass

    @abstractmethod
    def save_model(self, model_path: str) -> None:
        pass

    @property
    @abstractmethod
    def model(self) -> Model:
        pass

    @property
    @abstractmethod
    def steps_per_epoch(self) -> int:
        pass


class FittingDataset:

    def __init__(self, tf_dataset: TFDataset, batches_per_epoch: int):
        self.tf_dataset = tf_dataset
        self.batches_per_epoch = batches_per_epoch


class DefaultTrainer(TrainerInterface):

    def __init__(
            self, model: Model, trainer_heads: List[TrainerHeadInterface],
            fitting_training_set: FittingDataset,
            fitting_validation_set: Optional[FittingDataset]) -> None:
        self._model = model
        self._trainer_heads = trainer_heads
        # Using fit with the Tensorflow Dataset was the fastest solution
        # I could find, so we use that here. It does require converting some
        # of the data pre-processing components to Tensorflow, which can be
        # annoying sometimes.
        self._fitting_training_set = fitting_training_set
        self._fitting_validation_set = fitting_validation_set

    def compile(self, optimizer: Optimizer) -> None:
        losses = [
            trainer_head.predictor_head.loss
            for trainer_head in self._trainer_heads]
        loss_weights = [
            trainer_head.predictor_head.loss_weight
            for trainer_head in self._trainer_heads]
        # Keras has some memory leaks that we are trying to work around. When
        # using run_eagerly=True, the leak is smaller. In theory,
        # it is slower, but I didn't notice a significant decrease in speed
        # in some experiments.
        self._model.compile(
            loss=losses, optimizer=optimizer, loss_weights=loss_weights,
            run_eagerly=True)

    def fit(self, initial_epoch: int, epochs: int, callbacks,
            validation_freq) -> History:
        if self._fitting_validation_set:
            validation_tf_dataset = self._fitting_validation_set.tf_dataset
        else:
            validation_tf_dataset = None
        return self._model.fit(
            self._fitting_training_set.tf_dataset,
            initial_epoch=initial_epoch,
            epochs=epochs, callbacks=callbacks,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=validation_tf_dataset,
            validation_freq=validation_freq,
            validation_steps=self.validation_steps)

    def save_model(self, model_path: str) -> None:
        self._model.save(model_path)

    @property
    def model(self) -> Model:
        return self._model

    @property
    def steps_per_epoch(self) -> int:
        return self._fitting_training_set.batches_per_epoch

    @property
    def validation_steps(self) -> Optional[int]:
        if not self._fitting_validation_set:
            return None
        return self._fitting_validation_set.batches_per_epoch
