# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import datetime
import gc
import logging
import os
import pickle
import warnings
from typing import Dict, Any, Optional

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.summary import SummaryWriter

from spivak.application.argument_parser import LEARNING_RATE_DECAY_LEGACY, \
    SharedArgs
from spivak.application.dataset_creation import create_label_maps, \
    create_datasets
from spivak.application.model_creation import load_or_create_trainer
from spivak.application.validation import LAST_MODEL_DIR, ValidationResult, \
    KERAS_GENERIC_UTILS
from spivak.application.worker_manager import Manager, ChildTask
from spivak.data.dataset_splits import SPLIT_KEY_TRAIN, SPLIT_KEY_VALIDATION
from spivak.models.trainer import TrainerInterface

PROFILE_BATCH_DISABLE = 0
PROFILE_BATCH_TRIO = [50, 52]
PROFILE_BATCH = PROFILE_BATCH_DISABLE  # or PROFILE_BATCH_TRIO
FIX_ITERATIONS_EPOCHS = 10
MIN_SAVE_EPOCHS = 5
METRIC_ITERATIONS = "iterations"
METRIC_LEARNING_RATE = "learning_rate"
TRAINING_STATE_FILE = "last_training_state.pkl"
LEGACY_FINAL_LEARNING_RATE = 10 ** -6
# Setting the memory limit instead of allowing growth allowed for larger
# batch sizes, while still being able to run the validation process
# separately. Using memory growth also works well in general, but for some
# reason it would take up too much GPU memory when using large batch sizes,
# not allowing the validation process to run.
MEMORY_GROWTH = False
# About 20GB
MEMORY_LIMIT_IN_MB = 20 * 1024
KERAS_MIXED_PRECISION_POLICY = "mixed_float16"


def train(args: SharedArgs, manager: Manager) -> None:
    if args.save_epochs < MIN_SAVE_EPOCHS:
        raise ValueError(
            f"save_epochs argument should be at least {MIN_SAVE_EPOCHS} in "
            f"order to give time for the validation process to read the last "
            f"model, before overwriting it.")
    # Set logging levels
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(action="ignore", module=KERAS_GENERIC_UTILS)
    _set_mixed_precision(bool(args.mixed_precision))
    # Load the datasets
    logging.info("Preparing datasets")
    label_maps = create_label_maps(args)
    training_set, validation_set = create_datasets(
        args, [SPLIT_KEY_TRAIN, SPLIT_KEY_VALIDATION], label_maps)
    if validation_set:
        # Adjust GPU memory, so we have some memory left over in order to run
        # validation in another process.
        _adjust_gpu_memory()
    else:
        # Ask the manager of the validation process to exit, since validation
        # won't be used in this run.
        manager.input_queue.put(ChildTask(do_exit=True, args=None))
    # Load the training state.
    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)
    training_state = TrainingState.load_or_create(model_dir)
    logging.info(training_state)
    last_model_path = os.path.join(model_dir, LAST_MODEL_DIR)
    # Load the trainer.
    trainer = load_or_create_trainer(
        last_model_path, args, training_set, validation_set, label_maps)
    trainer.model.summary()
    # Set up tensorboard and summary_writer for saving training and validation
    # metrics to disk.
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_log_dir = os.path.join(model_dir, 'logs', current_time)
    train_tensorboard = _create_tensorboard(
        trainer.model, "epoch", os.path.join(base_log_dir, "train"))
    summary_writer = tf.summary.create_file_writer(
        os.path.join(base_log_dir, "metrics"))
    summary_writer.set_as_default()
    # Save the arguments for this run inside the log dir.
    args.save(base_log_dir)
    # Create legacy learning rate decay class (only applied if needed).
    should_apply_decay = (args.learning_rate_decay == LEARNING_RATE_DECAY_LEGACY)
    legacy_learning_rate_decay = LegacyLearningRateDecay(
        trainer.model, args.learning_rate, args.epochs, should_apply_decay)
    # Create training and validation callbacks.
    training_callback = TrainingCallback(
        args, training_state, trainer, summary_writer, args.save_epochs,
        model_dir, last_model_path, legacy_learning_rate_decay)
    # The order of the callbacks below is usually important and should be
    # preserved unless there are changes to the callback definitions.
    callbacks = [training_callback]
    if validation_set:
        validation_callback = ValidationCallback(
            manager, summary_writer, args, training_state, last_model_path,
            args.validation_epochs)
        callbacks.append(validation_callback)
    callbacks.append(train_tensorboard)
    # There appears to be some memory leak in Keras fit() when using
    # tensorflow.data.Dataset. It seems to help to call it only once.
    trainer.fit(
        training_state.epoch, args.epochs, callbacks, args.validation_epochs)


class TrainingState:

    def __init__(
            self, epoch: int, iterations: int, best_metric: float,
            metric_name: Optional[str], best_epoch: int) -> None:
        self.epoch = epoch
        self.iterations = iterations
        # For saving the best results
        self.best_metric = best_metric
        self.metric_name = metric_name
        self.best_epoch = best_epoch
        # For some reason, I can't change the optimizer iterations directly
        # after loading it from disk, otherwise its state gets reset somehow and
        # the loss always jumps up. Instead, if the optimizer iterations is
        # wrong, we first let it run for some epochs and then fix it.
        # broken_iterations_epochs counts how many epochs the optimizer
        # iterations count has been broken for.
        self.broken_iterations_epochs = 0

    def save(self, save_path: str) -> None:
        training_state_path = os.path.join(save_path, TRAINING_STATE_FILE)
        with open(training_state_path, "wb") as pickle_file:
            return pickle.dump(self, pickle_file)

    def __str__(self):
        if self.metric_name:
            current_metric_name = self.metric_name
        else:
            current_metric_name = "none"
        return (f"TrainingState {{epoch: {self.epoch}, "
                f"iterations: {self.iterations}, "
                f"best_metric ({current_metric_name}): {self.best_metric}, "
                f"best_epoch: {self.best_epoch}, "
                f"broken_iterations_epochs: {self.broken_iterations_epochs}}}")

    @staticmethod
    def load_or_create(save_path: str) -> "TrainingState":
        training_state_path = os.path.join(save_path, TRAINING_STATE_FILE)
        if not os.path.exists(training_state_path):
            return TrainingState(0, 0, 0.0, None, 0)
        with open(training_state_path, "rb") as pickle_file:
            training_state: TrainingState = pickle.load(pickle_file)
            # It's probably better to reset the counter here to zero in case
            # there was some count accumulated from a previous run.
            training_state.broken_iterations_epochs = 0
            return training_state


class TrainingCallback(Callback):

    def __init__(
            self, args: SharedArgs, training_state: TrainingState,
            trainer: TrainerInterface, summary_writer: SummaryWriter,
            save_frequency: int, save_path: str, last_model_path: str,
            legacy_learning_rate_decay: "LegacyLearningRateDecay"):
        super().__init__()
        self._args = args
        self._training_state = training_state
        self._trainer = trainer
        self._summary_writer = summary_writer
        self._save_frequency = save_frequency
        self._save_path = save_path
        self._last_model_path = last_model_path
        self._legacy_learning_rate_decay = legacy_learning_rate_decay

    def on_epoch_end(self, epoch, logs=None):
        _update_iterations(
            self._training_state, self._trainer.model,
            self._trainer.steps_per_epoch, epoch)
        # Do legacy learning rate decay if needed.
        self._legacy_learning_rate_decay.apply_if_needed()
        learning_rate = _get_learning_rate(
            self._trainer.model, self._training_state.iterations)
        scalars_to_write = {
            METRIC_LEARNING_RATE: learning_rate,
            METRIC_ITERATIONS: self._training_state.iterations
        }
        _write_scalars(self._summary_writer, scalars_to_write, epoch)
        self._training_state.epoch = epoch
        if (self._save_frequency and epoch != 0
                and epoch % self._save_frequency == 0):
            self._trainer.save_model(self._last_model_path)
            self._args.save(self._last_model_path)
            self._training_state.save(self._save_path)
        _garbage_collect()


class ValidationCallback(Callback):

    """The validation runs in a separate process to mitigate the related memory
    leak."""

    def __init__(
            self, manager: Manager, summary_writer: SummaryWriter, args,
            training_state: "TrainingState", last_model_path,
            validation_frequency):
        super().__init__()
        self._args = args
        self._validation_frequency = validation_frequency
        self._training_state = training_state
        self._last_model_path = last_model_path
        self._summary_writer = summary_writer
        self._manager = manager
        self._task_is_pending = False

    def on_epoch_end(self, epoch, logs=None):
        if (self._validation_frequency and epoch != 0
                and epoch % self._validation_frequency == 0):
            self._collect_results_and_submit_validation_task(epoch)

    def on_train_end(self, logs=None):
        self._collect_results_if_pending()
        self._manager.input_queue.put(ChildTask(do_exit=True, args=None))

    def _collect_results_and_submit_validation_task(self, epoch):
        # This process will read the latest models and its state.
        self._collect_results_if_pending()
        if os.path.exists(self._last_model_path):
            self._manager.input_queue.put(ChildTask(
                do_exit=False,
                args=(self._args, self._training_state.best_metric, epoch)
            ))
            self._task_is_pending = True

    def _collect_results_if_pending(self):
        if self._task_is_pending:
            self._collect_results()

    def _collect_results(self):
        validation_result: ValidationResult = self._manager.output_queue.get()
        evaluation = validation_result.evaluation
        self._task_is_pending = False
        if evaluation.main_metric > self._training_state.best_metric:
            self._training_state.best_metric = evaluation.main_metric
            self._training_state.metric_name = evaluation.main_metric_name
            self._training_state.best_epoch = validation_result.epoch
        scalars_to_write = evaluation.scalars_for_logging()
        _write_scalars(
            self._summary_writer, scalars_to_write, validation_result.epoch)


class LegacyLearningRateDecay:

    def __init__(self, model, initial_learning_rate, epochs, should_apply):
        self._model = model
        self._initial_learning_rate = initial_learning_rate
        self._epochs = epochs
        self._should_apply = should_apply

    def apply_if_needed(self):
        if self._should_apply:
            _legacy_decay_learning_rate(
                self._model, self._initial_learning_rate, self._epochs)


def _set_mixed_precision(should_set_mixed_precision: bool) -> None:
    if should_set_mixed_precision:
        # I think this API is only available in newer versions of Tensorflow.
        tf.keras.mixed_precision.set_global_policy(KERAS_MIXED_PRECISION_POLICY)


def _adjust_gpu_memory():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        if MEMORY_GROWTH:
            tf.config.experimental.set_memory_growth(gpu, True)
        else:
            device_config = tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=MEMORY_LIMIT_IN_MB)
            tf.config.experimental.set_virtual_device_configuration(
                gpu, [device_config])


def _create_tensorboard(model: Model, update_freq, log_dir: str):
    tensorboard = tensorflow.keras.callbacks.TensorBoard(
       log_dir, update_freq=update_freq, profile_batch=PROFILE_BATCH)
    tensorboard.set_model(model)
    return tensorboard


def _get_learning_rate(network: Model, step: int):
    if isinstance(network.optimizer.lr, LearningRateSchedule):
        lr_tensor = network.optimizer.lr(step)
    else:
        lr_tensor = network.optimizer.lr
    return K.get_value(lr_tensor)


def _legacy_decay_learning_rate(
        network: Model, initial_learning_rate: float, epochs: int) -> None:
    current_learning_rate = K.get_value(network.optimizer.lr)
    new_learning_rate = (
            current_learning_rate
            - (initial_learning_rate - LEGACY_FINAL_LEARNING_RATE) / epochs
    )
    K.set_value(network.optimizer.lr, new_learning_rate)


def _update_iterations(
        training_state: TrainingState, network: Model,
        steps_per_epoch: int, epoch: int) -> None:
    current_iterations = K.get_value(network.optimizer.iterations)
    if current_iterations > training_state.iterations:
        training_state.iterations = current_iterations
    else:
        # This is probably because we just loaded a models from disk, whose
        # optimizer iterations got reset to zero.
        training_state.iterations += steps_per_epoch
        training_state.broken_iterations_epochs += 1
        # We can't immediately set the optimizer iterations to the
        # correct value due to some weird Keras behavior where the loss
        # jumps if we do. As a workaround, we wait for some iterations
        # and then set the iterations to the desired value here.
        if training_state.broken_iterations_epochs >= FIX_ITERATIONS_EPOCHS:
            logging.warning("Fixing optimizer iterations.")
            K.set_value(network.optimizer.iterations, training_state.iterations)
            training_state.broken_iterations_epochs = 0


def _write_scalars(
        summary_writer: SummaryWriter, scalars: Dict[str, Any],
        epoch: int) -> None:
    with summary_writer.as_default():
        for scalar_name in scalars:
            tf.summary.scalar(
                scalar_name, data=scalars[scalar_name], step=epoch)


def _garbage_collect():
    # This helps reduce memory to work around the memory leaks. It's
    # important to call this frequently, otherwise it doesn't work very
    # well.
    tf.keras.backend.clear_session()
    gc.collect()
