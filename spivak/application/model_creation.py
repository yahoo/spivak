# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging
import math
import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import tensorflow.keras
import tensorflow_addons as tfa
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import get_custom_objects

from spivak.application.argument_parser import SharedArgs, DETECTOR_DENSE, \
    DETECTOR_DENSE_DELTA, DETECTOR_AVERAGING_CONFIDENCE, \
    DETECTOR_AVERAGING_DELTA, LEARNING_RATE_DECAY_LEGACY, \
    LEARNING_RATE_DECAY_EXPONENTIAL, LEARNING_RATE_DECAY_LINEAR, \
    OPTIMIZER_ADAM, OPTIMIZER_SGD, dir_str_to_path, SAMPLING_UNIFORM, \
    SAMPLING_WEIGHTED, BACKBONE_UNET, BACKBONE_TRANSFORMER_ENCODER, \
    UNET_COMBINER_CONCATENATION, UNET_COMBINER_ADDITION, \
    UNET_UPSAMPLER_TRANSPOSE, UNET_UPSAMPLER_UPSAMPLING, UNET_COMBINER_IGNORE, \
    NMS_DECAY_SUPPRESS, NMS_DECAY_LINEAR, NMS_DECAY_GAUSSIAN
from spivak.data.dataset import Dataset, VideoDatum, InputShape, Task
from spivak.data.label_map import LabelMap
from spivak.data.output_names import OUTPUT_SEGMENTATION
from spivak.data.video_chunk_iterator import VideoChunkIteratorProvider
from spivak.models.assembly.bottom_up import create_bottom_up_layers, \
    StridedBottomUpStack, PoolingBottomUpStack, BottomUpLayer, POOLING_MAX, \
    POOLING_AVERAGE
from spivak.models.assembly.confidence_head import ConfidenceTrainerHead, \
    ConfidenceAuxTrainerHead, ConfidenceWeightCreator, \
    ConfidencePredictorHead, ConfidenceAuxPredictorHead, \
    ConfidenceLoss, ConfidenceAuxLoss, ConfidenceClassCounts
from spivak.models.assembly.convolution_stacks import BasicConvolutionStack, \
    ResidualBlock, ResidualBlockV2, StridedBlockInterface
from spivak.models.assembly.cyclic_delta_head import CyclicDeltaPredictorHead, \
    CyclicDeltaLoss
from spivak.models.assembly.delta_aux_head import DeltaAuxTrainerHead, \
    DeltaAuxPredictorHead, CyclicDeltaAuxLoss
from spivak.models.assembly.delta_head import DeltaTrainerHead, \
    create_huber_base_loss, DeltaWeightCreator, DeltaPredictorHeadInterface, \
    DeltaLoss, DeltaPredictorHead
from spivak.models.assembly.head import TrainerHeadInterface, \
    PredictorHeadInterface
from spivak.models.assembly.layers import create_main_input, input_reduction, \
    Nodes, NODE_OPTIONAL_PROJECTION
from spivak.models.assembly.segmentation_head import SegmentationTrainerHead, \
    SegmentationPredictorHead, SegmentationLoss, SegmentationWeightCreator
from spivak.models.assembly.transformer_encoder_backbone import \
    TransformerEncoderConfig, create_transformer_encoder_backbone
from spivak.models.assembly.unet_backbone import ConcatenationCombiner, \
    AdditionCombiner, ConvTransposeUpsampler, UpsamplingUpsampler, \
    BasicTopDownStack, create_unet_backbone, TopDownStackInterface, \
    IgnoreCombiner
from spivak.models.assembly.weight_creator import IdentityWeightCreator, \
    SampledWeightCreator, read_class_weights
# I think these classes were needed so that we're able to load the
# corresponding pickle files later below.
# noinspection PyUnresolvedReferences
from spivak.models.averaging_predictor import DeltaAveragingPredictor, \
    ConfidenceAveragingPredictor
from spivak.models.delta_dense_predictor import DeltaDensePredictor
from spivak.models.dense_predictor import OUTPUT_CONFIDENCE, OUTPUT_DELTA, \
    OUTPUT_CONFIDENCE_AUX, OUTPUT_DELTA_AUX, DensePredictor
from spivak.models.non_maximum_suppression import \
    FlexibleNonMaximumSuppression, ScoreDecaySuppress, ScoreDecayLinear
from spivak.models.predictor import PredictorInterface
from spivak.models.projector import Projector
from spivak.models.sam_model import SAMModel, maybe_convert_model_to_sam
from spivak.models.tf_dataset import create_tf_merged_batch_dataset, \
    create_tf_get_video_chunks, create_tf_mixup_batch_augmentation, TFDataset, \
    create_tf_task_batch_dataset, create_tf_task_videos_datasets
from spivak.models.trainer import DefaultTrainer, FittingDataset, \
    TrainerInterface
from spivak.models.video_start_provider import VideoStartProviderInterface, \
    VideoStartProviderUniform, VideoStartProviderWeighted, \
    compute_min_valid_chunk_frames, StartProbabilitiesCreator

NMS_WINDOWS_CSV = "nms_windows.csv"
CLASS_WEIGHTS_CSV_SPOTTING = "spotting_class_weights.csv"
CLASS_WEIGHTS_CSV_SEGMENTATION = "segmentation_class_weights.csv"
END_LEARNING_RATE_FRACTION = 0.01
SAM_EPSILON = 1e-12
DENSE_HEAD_NAMES = [
    OUTPUT_CONFIDENCE, OUTPUT_DELTA, OUTPUT_CONFIDENCE_AUX, OUTPUT_DELTA_AUX,
    OUTPUT_SEGMENTATION]
# When training the deltas by themselves (without the confidences in the same
# models), initializing with zeros makes optimization much harder, so it's
# better not to use it. However, when training with the confidences, sometimes
# the zero initialization can help, possibly because we might not be training
# the deltas for long enough, so that it's better to output something close
# to zero. In practice, even when training with the confidences, the result is
# usually not very different whether using zero initialization or not.
DELTA_ZERO_INITIALIZE = False
# These DELTA_AUX_RADII_IN_SECONDS were determined experimentally.
# Experiments were limited, but using the values below might give a slight
# benefit over just using [30.0]. It also seems to optimize faster.
DELTA_AUX_RADII_IN_SECONDS = [15.0, 30.0, 60.0, 90.0]
# 20.0 seems to be a good value based on experiments on SoccerNet v1.
CONFIDENCE_AUX_RADII_IN_SECONDS = [20.0]
INDEFINITE_REPETITIONS = None
CHUNKS_PER_EPOCH = 8192
MIN_ACCEPTED_PREDICTION_SECONDS = 10.0
# Backbones
# U-net parameters
UNET_MAX_LAYERS = 10
UNET_MAX_FILTERS = 2048
UNET_RESIDUAL_V2 = True
# Bottleneck blocks seem to work as well as non-bottleneck blocks,
# while using significantly less parameters.
UNET_BLOCK_BOTTLENECK = True
UNET_BLOCK_NUM_CONVOLUTIONS = 1
# Using more blocks per layer did not help.
UNET_BOTTOM_UP_STACK_NUM_BLOCKS = [2] * UNET_MAX_LAYERS
UNET_TOP_DOWN_STACK_NUM_BLOCKS = [2] * UNET_MAX_LAYERS
UNET_DOWNSAMPLE_POOLING_MAX = POOLING_MAX
UNET_DOWNSAMPLE_POOLING_AVERAGE = POOLING_AVERAGE
UNET_DOWNSAMPLE_STRIDE = "stride"
# In early experiments, max-pooling down-sampling did a bit better than
# strided convolution and average pooling. Might need to redo these at some
# point with new models changes.
UNET_DOWNSAMPLE = UNET_DOWNSAMPLE_POOLING_MAX
UNET_ACTIVATION = "relu"
UNET_UPSAMPLING_CONVOLVE = False
UNET_UPSAMPLING_INTERPOLATION = "nearest"
# In a single initial experiment, False was better here.
UNET_ADDITION_COMBINER_PRE_CONVOLVE = False
# Transformer encoder backbone parameters.
# Learned positional encodings seemed to work better overall. Sinusoidal
# positional embeddings seemed to work almost as well as learned positional
# embeddings, so they could be an option to consider as well. They are not as
# dependent on learning, and are better at handling varying sequence lengths.
# Learned convolutional positional embeddings performed worse than the
# learned vanilla and the sinusoidal one.
TRANSFORMER_POSITIONAL_EMBEDDING = \
    TransformerEncoderConfig.LEARNED_POSITIONAL_EMBEDDING
TRANSFORMER_CONVOLUTIONAL_POSITION_EMBEDDING_ACTIVATION = "gelu"
# 128 is a weird value, but it is copied from original code:
# https://github.com/pytorch/fairseq/blob/eb2bed115497834a800ff787d36d6615205462d0/fairseq/models/wav2vec/wav2vec2.py#L217
# https://github.com/pytorch/fairseq/blob/eb2bed115497834a800ff787d36d6615205462d0/fairseq/models/wav2vec/wav2vec2.py#L827
TRANSFORMER_CONVOLUTIONAL_POSITIONAL_EMBEDDING_KERNEL = 128
TRANSFORMER_CONVOLUTIONAL_POSITIONAL_EMBEDDING_GROUPS = 16
# In early experiments, the stable layer norm seems to be slightly helpful.
TRANSFORMER_STABLE_LAYER_NORM = True
# Tweaked the layer norm eps a bit. Original value of 1e-5 also did well.
TRANSFORMER_LAYER_NORM_EPS = 1e-6
TRANSFORMER_HIDDEN_ACTIVATION = "gelu"
TRANSFORMER_INITIALIZER_RANGE = 0.02
TRANSFORMER_LAYER_DROPOUT = 0.0
TRANSFORMER_HIDDEN_DROPOUT = 0.0
TRANSFORMER_ACTIVATION_DROPOUT = 0.0
TRANSFORMER_ATTENTION_DROPOUT = 0.0
TRANSFORMER_ENCODER_NAME = "encoder"
NMS_LINEAR_EXPANSION = 2.0
PROJECTION_LAYER_NAME = f"{NODE_OPTIONAL_PROJECTION}_relu"


def create_flexible_nms(
        args: SharedArgs, label_map: LabelMap
) -> FlexibleNonMaximumSuppression:
    config_dir = dir_str_to_path(args.config_dir)
    potential_nms_windows_path = config_dir / NMS_WINDOWS_CSV
    if potential_nms_windows_path.exists():
        class_windows_in_seconds = \
            FlexibleNonMaximumSuppression.read_nms_windows(
                potential_nms_windows_path, label_map)
        class_windows = args.frame_rate * class_windows_in_seconds
    else:
        # The unusual calculation of "window" here follows the original
        # SoccerNet codebase. For comparison purposes, we prefer not to
        # change it here. This value below is used by the standard SoccerNet
        # code flow when saving results. However, during evaluation (v1), the
        # FlexibleNonMaximumSuppression class is not used and the NMS windows
        # come from the mAP tolerance being used in the evaluation (in v1
        # evaluation, for each mAP tolerance, a different NMS is applied with a
        # window size that matches the tolerance).
        window = 2 * int(args.nms_window * args.frame_rate / 2.0)
        class_windows = np.asarray([window] * label_map.num_classes())
    if args.nms_decay == NMS_DECAY_SUPPRESS:
        score_decay = ScoreDecaySuppress()
    elif args.nms_decay == NMS_DECAY_LINEAR:
        score_decay = ScoreDecayLinear(
            args.nms_decay_linear_min, NMS_LINEAR_EXPANSION)
    elif args.nms_decay == NMS_DECAY_GAUSSIAN:
        raise NotImplementedError(
            f"NMS decay {NMS_DECAY_GAUSSIAN} not implemented")
    else:
        raise ValueError(f"Unknown value for args.nmsdecay: {args.nms_decay}")
    return FlexibleNonMaximumSuppression(
        bool(args.apply_nms), class_windows, score_decay)


def load_or_create_trainer(
        last_model_path: str, args: SharedArgs, training_set: Dataset,
        validation_set: Optional[Dataset], label_maps: Dict[Task, LabelMap]
) -> TrainerInterface:
    # If models file already exists, just load it.
    if os.path.exists(last_model_path):
        if args.learning_rate_decay == LEARNING_RATE_DECAY_LEGACY:
            raise ValueError("Cannot load models to continue training when "
                             "using legacy learning rate decay")
        trainer = _load_trainer(
            args, last_model_path, label_maps, training_set, validation_set)
        if args.sam_rho > 0.0:
            # Due to monkey-patching the train_step function in the models for
            # SAM, we need to compile the models again. Hopefully, this won't
            # affect the state of the optimizer and models, but not sure.
            trainer.compile(trainer.model.optimizer)
        return trainer
    trainer = _create_trainer(args, label_maps, training_set, validation_set)
    if args.pretrained_path:
        trainer.model.load_weights(args.pretrained_path)
    optimizer = _optimizer(args, trainer.steps_per_epoch)
    trainer.compile(optimizer)
    return trainer


def load_predictor_from_model_path(
        args: SharedArgs, model_path: str,
        label_maps: Dict[Task, LabelMap], chunk_prediction_border: float
) -> PredictorInterface:
    return _load_predictor(
        args, model_path, label_maps, chunk_prediction_border)


def load_predictor(
        args: SharedArgs, label_maps: Dict[Task, LabelMap],
        input_shape: InputShape, chunk_prediction_border: float
) -> PredictorInterface:
    if args.model:
        return _load_predictor(
            args, args.model, label_maps, chunk_prediction_border)
    elif args.model_weights:
        predictor = _create_predictor(
            args, input_shape, label_maps, chunk_prediction_border)
        predictor.load_weights(args.model_weights)
        return predictor
    else:
        raise ValueError(
            "Neither weights nor models parameters were specified in the input "
            "arguments")


def load_projector(
        args: SharedArgs, label_maps: Dict[Task, LabelMap]
) -> Projector:
    model = _load_keras_model(args, label_maps)
    projector_model = Model(
        inputs=model.input,
        outputs=model.get_layer(PROJECTION_LAYER_NAME).output)
    video_chunk_iterator_provider = _video_chunk_iterator_provider(args, 0)
    return Projector(projector_model, video_chunk_iterator_provider)


def create_delta_radius(args: SharedArgs) -> float:
    # Parameterize delta with args.deltaradiusmultiplier * radius,
    # so we don't get wrap-around errors at around -pi and pi when we
    # are close to +/- radius. We will also use this same
    # args.deltaradiusmultiplier * radius for weights so we learn to
    # regress delta a bit outside the exact relevant region.
    radius = args.frame_rate * args.dense_detection_radius
    return args.delta_radius_multiplier * radius


def _load_trainer(
        args: SharedArgs, model_path: str,
        label_maps: Dict[Task, LabelMap], training_set: Dataset,
        validation_set: Optional[Dataset]) -> TrainerInterface:
    if args.detector in {DETECTOR_DENSE, DETECTOR_DENSE_DELTA}:
        trainer_heads = _dense_trainer_heads(args, label_maps, training_set)
        predictor_heads = [
            trainer_head.predictor_head for trainer_head in trainer_heads]
        _update_keras_custom_objects(predictor_heads)
        model = _load_model(args, model_path)
        return _trainer(
            model, trainer_heads, args, training_set, validation_set)
    else:
        raise ValueError(f"Unknown detector choice: {args.detector}")


def _create_trainer(
        args: SharedArgs, label_maps: Dict[Task, LabelMap],
        training_set: Dataset, validation_set: Optional[Dataset]
) -> TrainerInterface:
    input_shape = training_set.input_shape
    if args.detector in {DETECTOR_DENSE, DETECTOR_DENSE_DELTA}:
        trainer_heads = _dense_trainer_heads(args, label_maps, training_set)
        predictor_heads = [
            trainer_head.predictor_head for trainer_head in trainer_heads]
        model = _create_keras_model(args, input_shape, predictor_heads)
        return _trainer(
            model, trainer_heads, args, training_set, validation_set)
    else:
        raise ValueError(f"Unknown detector choice: {args.detector}")


def _load_predictor(
        args: SharedArgs, model_path: str,
        label_maps: Dict[Task, LabelMap], chunk_prediction_border: float
) -> PredictorInterface:
    if args.detector == DETECTOR_DENSE:
        return _load_dense_predictor(
            args, model_path, label_maps, chunk_prediction_border)
    elif args.detector == DETECTOR_DENSE_DELTA:
        dense_predictor = _load_dense_predictor(
            args, model_path, label_maps, chunk_prediction_border)
        return DeltaDensePredictor(
            dense_predictor, dir_str_to_path(args.results_dir))
    elif args.detector in {DETECTOR_AVERAGING_CONFIDENCE,
                           DETECTOR_AVERAGING_DELTA}:
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    else:
        raise ValueError(f"Unknown detector choice: {args.detector}")


def _load_dense_predictor(
        args: SharedArgs, model_path: str,
        label_maps: Dict[Task, LabelMap], chunk_prediction_border: float
) -> DensePredictor:
    predictor_heads = _dense_predictor_heads(args, label_maps)
    _update_keras_custom_objects(predictor_heads)
    model = _load_model(args, model_path)
    return _dense_predictor(
        model, predictor_heads, args, chunk_prediction_border)


def _create_predictor(
        args: SharedArgs, input_shape: InputShape,
        label_maps: Dict[Task, LabelMap], chunk_prediction_border: float
) -> PredictorInterface:
    if args.detector == DETECTOR_DENSE:
        return _create_dense_predictor(
            args, input_shape, label_maps, chunk_prediction_border)
    elif args.detector == DETECTOR_DENSE_DELTA:
        dense_predictor = _create_dense_predictor(
            args, input_shape, label_maps, chunk_prediction_border)
        return DeltaDensePredictor(
            dense_predictor, dir_str_to_path(args.results_dir))
    elif args.detector in {DETECTOR_AVERAGING_CONFIDENCE,
                           DETECTOR_AVERAGING_DELTA}:
        raise ValueError(
            f"Creation of {args.detector} not supported. Please use the "
            f"create_averaging_predictor.py script instead.")
    else:
        raise ValueError(f"Unknown detector choice: {args.detector}")


def _create_dense_predictor(
        args: SharedArgs, input_shape: InputShape,
        label_maps: Dict[Task, LabelMap], chunk_prediction_border: float
) -> DensePredictor:
    predictor_heads = _dense_predictor_heads(args, label_maps)
    model = _create_keras_model(args, input_shape, predictor_heads)
    return _dense_predictor(
        model, predictor_heads, args, chunk_prediction_border)


def _trainer(
        model: Model, trainer_heads: List[TrainerHeadInterface],
        args: SharedArgs, training_set: Dataset,
        validation_set: Optional[Dataset]) -> DefaultTrainer:
    logging.info(f"Preparing training TF dataset from "
                 f"{training_set.num_videos} videos")
    fitting_training_set = _fitting_dataset(
        args, training_set, trainer_heads, INDEFINITE_REPETITIONS,
        shuffle_videos=args.shuffle_videos, chunk_shuffle=args.chunk_shuffle)
    if validation_set:
        logging.info(f"Preparing validation TF dataset from "
                     f"{validation_set.num_videos} videos")
        # TODO: maybe remove randomness when generating batches for validation.
        fitting_validation_set = _fitting_dataset(
            args, validation_set, trainer_heads, INDEFINITE_REPETITIONS,
            shuffle_videos=False, chunk_shuffle=0.0)
    else:
        fitting_validation_set = None
    return DefaultTrainer(
        model, trainer_heads, fitting_training_set, fitting_validation_set)


def _fitting_dataset(
        args: SharedArgs, dataset: Dataset,
        heads: List[TrainerHeadInterface], repetitions: Optional[int],
        shuffle_videos: bool, chunk_shuffle: float) -> FittingDataset:
    # At the moment, we could initialize video_start_providers outside and
    # share it between the training and validation datasets, but eventually
    # those datasets should have different video_start_provider settings.
    video_start_providers = _video_start_providers(args, dataset)
    tf_dataset = _tf_dataset(
        args, dataset, heads, video_start_providers, repetitions,
        shuffle_videos, chunk_shuffle)
    num_tasks = len(dataset.tasks)
    num_batches_per_epoch = num_tasks * math.ceil(
        CHUNKS_PER_EPOCH / args.batch_size)
    return FittingDataset(tf_dataset, num_batches_per_epoch)


def _tf_dataset(
        args: SharedArgs, dataset: Dataset,
        heads: List[TrainerHeadInterface],
        video_start_providers: Dict[Task, VideoStartProviderInterface],
        repetitions: Optional[int], shuffle_videos: bool,
        chunk_shuffle: float) -> TFDataset:
    num_chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    if args.mixup_alpha:
        mixup_batch_augmentation = create_tf_mixup_batch_augmentation(
            heads, args.mixup_alpha)
    else:
        mixup_batch_augmentation = None
    batch_datasets = []
    tf_task_videos_datasets = create_tf_task_videos_datasets(
        dataset, heads, video_start_providers, bool(args.cache_dataset),
        shuffle_videos)
    for task in tf_task_videos_datasets:
        tf_get_video_chunks = create_tf_get_video_chunks(
            heads, task, num_chunk_frames)
        video_start_provider = video_start_providers[task]
        chunk_shuffle_size_float = \
            video_start_provider.get_num_chunks_dataset_float(
                dataset.video_data) * chunk_shuffle
        chunk_shuffle_size = math.ceil(chunk_shuffle_size_float)
        if chunk_shuffle_size:
            logging.info(f"chunk_shuffle_size: {chunk_shuffle_size}")
        task_batch_dataset = create_tf_task_batch_dataset(
            tf_task_videos_datasets[task], tf_get_video_chunks,
            args.get_chunks_parallelism, repetitions, args.batch_size,
            chunk_shuffle_size, mixup_batch_augmentation)
        batch_datasets.append(task_batch_dataset)
    return create_tf_merged_batch_dataset(batch_datasets)


def _video_start_providers(
        args: SharedArgs, dataset: Dataset
) -> Dict[Task, VideoStartProviderInterface]:
    return {
        task: _video_start_provider(args, dataset.video_data, task)
        for task in dataset.tasks
    }


def _video_start_provider(
        args: SharedArgs, video_data: List[VideoDatum], task: Task
) -> VideoStartProviderInterface:
    num_chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    min_valid_chunk_frames = compute_min_valid_chunk_frames(
        video_data, args.frame_rate, args.min_valid_chunk_duration)
    if args.sampling == SAMPLING_UNIFORM:
        return VideoStartProviderUniform(
            args.chunks_per_minute, min_valid_chunk_frames, args.frame_rate,
            task)
    elif args.sampling == SAMPLING_WEIGHTED:
        start_probabilities_creator = StartProbabilitiesCreator(
            num_chunk_frames, min_valid_chunk_frames,
            args.sampling_negative_fraction)
        return VideoStartProviderWeighted(
            args.chunks_per_minute, args.frame_rate,
            start_probabilities_creator)
    else:
        raise ValueError(f"Unknown negative sampling strategy: {args.sampling}")


def _dense_predictor(
        model: Model, predictor_heads: List[PredictorHeadInterface],
        args: SharedArgs, chunk_prediction_border: float
) -> DensePredictor:
    video_chunk_iterator_provider = _video_chunk_iterator_provider(
        args, chunk_prediction_border)
    return DensePredictor(
        model, predictor_heads, video_chunk_iterator_provider,
        bool(args.throw_out_delta), bool(args.profile))


def _optimizer(args: SharedArgs, steps_per_epoch: int) -> Optimizer:
    # Create the learning rate or schedule. Following the paper "Cyclical
    # Learning Rates for Training Neural Networks", the schedule is based on
    # epochs instead of on steps. It is also easier to debug and tune when
    # using epochs. There might be some advantages of using steps instead of
    # epochs, but I'm not sure.
    learning_rate = _create_learning_rate(args, steps_per_epoch)
    # Create the optimizer
    if args.decoupled_weight_decay:
        weight_decay = _create_weight_decay(args, steps_per_epoch)
        if args.optimizer == OPTIMIZER_ADAM:
            optimizer = tfa.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay,
                beta_1=0.9, beta_2=0.999, amsgrad=False, epsilon=1e-7)
        elif args.optimizer == OPTIMIZER_SGD:
            optimizer = tfa.optimizers.SGDW(
                learning_rate=learning_rate, weight_decay=weight_decay,
                momentum=0.9)
        else:
            raise ValueError(f"Unrecognized optimizer: {args.optimizer}")
    else:
        if args.optimizer == OPTIMIZER_ADAM:
            optimizer = tensorflow.keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                amsgrad=False, epsilon=1e-7)
        elif args.optimizer == OPTIMIZER_SGD:
            optimizer = tensorflow.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unrecognized optimizer: {args.optimizer}")
    return optimizer


def _create_learning_rate(args: SharedArgs, steps_per_epoch: int):
    return _maybe_create_schedule(args, steps_per_epoch, args.learning_rate)


def _create_weight_decay(args: SharedArgs, steps_per_epoch: int):
    # We'd like the weight decay schedule to follow the learning rate
    # schedule. See:
    # https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW
    # https://github.com/tensorflow/addons/issues/844
    return _maybe_create_schedule(
        args, steps_per_epoch, args.decoupled_weight_decay)


def _maybe_create_schedule(
        args: SharedArgs, steps_per_epoch: int, variable: float):
    if args.learning_rate_decay == LEARNING_RATE_DECAY_EXPONENTIAL:
        decay_steps = (
                args.learning_rate_decay_exponential_epochs * steps_per_epoch)
        variable = tensorflow.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=variable,
            decay_rate=args.learning_rate_decay_exponential_rate,
            decay_steps=decay_steps, staircase=False)
    elif args.learning_rate_decay == LEARNING_RATE_DECAY_LINEAR:
        decay_steps = args.learning_rate_decay_linear_epochs * steps_per_epoch
        end_variable = variable * END_LEARNING_RATE_FRACTION
        variable = tensorflow.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=variable, decay_steps=decay_steps,
            end_learning_rate=end_variable, power=1, cycle=True)
    return variable


def _load_keras_model(
        args: SharedArgs, label_maps: Dict[Task, LabelMap]) -> Model:
    if args.detector in {DETECTOR_DENSE, DETECTOR_DENSE_DELTA}:
        predictor_heads = _dense_predictor_heads(args, label_maps)
        _update_keras_custom_objects(predictor_heads)
        return _load_model(args, args.model)
    else:
        raise ValueError(f"Unsupported detector choice: {args.detector}")


def _dense_predictor_heads(
        args: SharedArgs, label_maps: Dict[Task, LabelMap]
) -> List[PredictorHeadInterface]:
    optional_heads = [
        _optional_predictor_head(args, head_name, label_maps)
        for head_name in DENSE_HEAD_NAMES]
    return [optional_head for optional_head in optional_heads
            if optional_head is not None]


def _optional_predictor_head(
        args: SharedArgs, head_name: str,
        label_maps: Dict[Task, LabelMap]) -> Optional[PredictorHeadInterface]:
    num_classes_spotting = label_maps[Task.SPOTTING].num_classes()
    if head_name == OUTPUT_CONFIDENCE:
        if args.confidence_weight == 0.0:
            return None
        return _confidence_predictor_head(
            args, head_name, num_classes_spotting, None)
    elif head_name == OUTPUT_DELTA:
        if args.delta_weight == 0.0:
            return None
        delta_radius = create_delta_radius(args)
        return _delta_predictor_head(
            args, head_name, num_classes_spotting, delta_radius)
    elif head_name == OUTPUT_CONFIDENCE_AUX:
        if args.confidence_aux_weight == 0.0:
            return None
        confidence_radii = [
            args.frame_rate * s for s in CONFIDENCE_AUX_RADII_IN_SECONDS]
        return _confidence_aux_predictor_head(
            args, head_name, num_classes_spotting, confidence_radii)
    elif head_name == OUTPUT_DELTA_AUX:
        if args.delta_aux_weight == 0.0:
            return None
        radii = [args.frame_rate * s for s in DELTA_AUX_RADII_IN_SECONDS]
        return _delta_aux_predictor_head(
            args, head_name, num_classes_spotting, radii)
    elif head_name == OUTPUT_SEGMENTATION:
        if args.segmentation_weight == 0.0:
            return None
        num_classes_segmentation = label_maps[Task.SEGMENTATION].num_classes()
        return _segmentation_predictor_head(
            args, head_name, num_classes_segmentation)
    else:
        raise ValueError(f"Unknown predictor head name: {head_name}")


def _dense_trainer_heads(
        args: SharedArgs, label_maps: Dict[Task, LabelMap],
        training_set: Dataset) -> List[TrainerHeadInterface]:
    optional_heads = [
        _optional_trainer_head(args, head_name, label_maps, training_set)
        for head_name in DENSE_HEAD_NAMES]
    return [optional_head for optional_head in optional_heads
            if optional_head is not None]


def _optional_trainer_head(
        args: SharedArgs, head_name: str,
        label_maps: Dict[Task, LabelMap], training_set: Dataset
) -> Optional[TrainerHeadInterface]:
    config_dir = dir_str_to_path(args.config_dir)
    spotting_class_weights = _read_spotting_class_weights(
        config_dir, label_maps[Task.SPOTTING])
    if head_name == OUTPUT_CONFIDENCE:
        if args.confidence_weight == 0.0:
            return None
        return _confidence_trainer_head(
            args, head_name, training_set, spotting_class_weights)
    elif head_name == OUTPUT_CONFIDENCE_AUX:
        if args.confidence_aux_weight == 0.0:
            return None
        return _confidence_aux_trainer_head(
            args, head_name, training_set, spotting_class_weights)
    elif head_name == OUTPUT_DELTA:
        if args.delta_weight == 0.0:
            return None
        return _delta_trainer_head(
            args, head_name, training_set, spotting_class_weights)
    elif head_name == OUTPUT_DELTA_AUX:
        if args.delta_aux_weight == 0.0:
            return None
        return _delta_aux_trainer_head(
            args, head_name, training_set, spotting_class_weights)
    elif head_name == OUTPUT_SEGMENTATION:
        if args.segmentation_weight == 0.0:
            return None
        segmentation_class_weights = _read_segmentation_class_weights(
            args.config_dir, label_maps[Task.SEGMENTATION])
        return _segmentation_trainer_head(
            args, head_name, training_set, segmentation_class_weights)
    else:
        raise ValueError(f"Unknown trainer head name: {head_name}")


def _read_spotting_class_weights(
        config_dir: Path, label_map: LabelMap) -> np.ndarray:
    class_weights_path = config_dir / CLASS_WEIGHTS_CSV_SPOTTING
    return read_class_weights(class_weights_path, label_map)


def _read_segmentation_class_weights(
        config_dir: Path, label_map: LabelMap) -> np.ndarray:
    class_weights_path = config_dir / CLASS_WEIGHTS_CSV_SEGMENTATION
    return read_class_weights(class_weights_path, label_map)


def _confidence_trainer_head(
        args: SharedArgs, head_name: str, training_set: Dataset,
        class_weights: np.ndarray) -> ConfidenceTrainerHead:
    # The convention here is that "confidence_radius" is already in frames,
    # so it is equivalent to the confidence_radius measured in seconds
    # multiplied by the frame rate.
    confidence_radius = args.frame_rate * args.dense_detection_radius
    confidence_class_counts = ConfidenceClassCounts(
        training_set, confidence_radius)
    if args.positive_weight_confidence:
        weight_creator = ConfidenceWeightCreator(
            confidence_class_counts, args.positive_weight_confidence,
            class_weights)
    else:
        sampled_weight_radius = 2 * confidence_radius
        weight_creator = SampledWeightCreator(
            sampled_weight_radius, args.sampling_negative_rate_confidence)
    if args.initialize_biases:
        predictor_class_counts = confidence_class_counts
    else:
        predictor_class_counts = None
    num_classes = len(class_weights)
    confidence_predictor_head = _confidence_predictor_head(
        args, head_name, num_classes, predictor_class_counts)
    return ConfidenceTrainerHead(
        confidence_radius, confidence_predictor_head, weight_creator)


def _confidence_aux_trainer_head(
        args: SharedArgs, head_name: str, training_set: Dataset,
        class_weights: np.ndarray) -> ConfidenceAuxTrainerHead:
    confidence_radii = [
        args.frame_rate * s for s in CONFIDENCE_AUX_RADII_IN_SECONDS]
    if args.positive_weight_confidence:
        weight_creators = [
            ConfidenceWeightCreator(
                ConfidenceClassCounts(training_set, radius),
                args.positive_weight_confidence, class_weights
            )
            for radius in confidence_radii
        ]
    else:
        identity_weight_creator = IdentityWeightCreator(class_weights)
        weight_creators = [
            identity_weight_creator for _ in confidence_radii]
    num_classes = len(class_weights)
    confidence_aux_predictor_head = _confidence_aux_predictor_head(
        args, head_name, num_classes, confidence_radii)
    return ConfidenceAuxTrainerHead(
        confidence_aux_predictor_head, weight_creators)


def _delta_trainer_head(
        args: SharedArgs, head_name: str, training_set: Dataset,
        class_weights: np.ndarray) -> DeltaTrainerHead:
    delta_radius = create_delta_radius(args)
    if args.positive_weight_delta:
        weight_creator = DeltaWeightCreator(
            training_set, delta_radius, args.positive_weight_delta,
            class_weights)
    else:
        weight_creator = SampledWeightCreator(
            delta_radius, args.sampling_negative_rate_delta)
    num_classes = len(class_weights)
    delta_predictor_head = _delta_predictor_head(
        args, head_name, num_classes, delta_radius)
    return DeltaTrainerHead(delta_predictor_head, weight_creator)


def _delta_aux_trainer_head(
        args: SharedArgs, head_name: str, training_set: Dataset,
        class_weights: np.ndarray) -> DeltaAuxTrainerHead:
    delta_radii = [args.frame_rate * s for s in DELTA_AUX_RADII_IN_SECONDS]
    if args.positive_weight_delta:
        weight_creators = [
            DeltaWeightCreator(
                training_set, radius, args.positive_weight_delta,
                class_weights)
            for radius in delta_radii
        ]
    else:
        identity_weight_creator = IdentityWeightCreator(class_weights)
        weight_creators = [identity_weight_creator for _ in delta_radii]
    num_classes = len(class_weights)
    delta_aux_predictor_head = _delta_aux_predictor_head(
        args, head_name, num_classes, delta_radii)
    return DeltaAuxTrainerHead(delta_aux_predictor_head, weight_creators)


def _delta_predictor_head(
        args: SharedArgs, head_name: str, num_classes: int,
        delta_radius: float) -> DeltaPredictorHeadInterface:
    num_chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    batch_norm = bool(args.batch_norm)
    # Did a single experiment, where Huber loss seems to be doing better on the
    # validation set than squared loss. Also, the detection loss jumps over time
    # when using squared, but not with Huber.
    huber_base_loss = create_huber_base_loss(args.huber_delta)
    if args.cyclic_delta:
        delta_loss = CyclicDeltaLoss(args.delta_weight, huber_base_loss)
        return CyclicDeltaPredictorHead(
            head_name, delta_radius, delta_loss, num_chunk_frames, num_classes,
            args.layer_weight_decay, batch_norm, args.dropout, args.width,
            args.head_layers, DELTA_ZERO_INITIALIZE)
    else:
        delta_loss = DeltaLoss(args.delta_weight, huber_base_loss)
        return DeltaPredictorHead(
            head_name, delta_radius, delta_loss, num_chunk_frames, num_classes,
            args.layer_weight_decay, batch_norm, args.dropout, args.width,
            args.head_layers, DELTA_ZERO_INITIALIZE)


def _delta_aux_predictor_head(
        args: SharedArgs, head_name: str, num_classes: int,
        radii: List[float]) -> DeltaAuxPredictorHead:
    huber_base_loss = create_huber_base_loss(args.huber_delta)
    delta_aux_loss = CyclicDeltaAuxLoss(
        args.delta_aux_weight, huber_base_loss, len(radii))
    batch_norm = bool(args.batch_norm)
    num_chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    return DeltaAuxPredictorHead(
        head_name, radii, delta_aux_loss, num_chunk_frames, num_classes,
        args.layer_weight_decay, batch_norm, args.dropout, args.width,
        args.head_layers, DELTA_ZERO_INITIALIZE)


def _confidence_predictor_head(
        args: SharedArgs, head_name: str, num_classes: int,
        confidence_class_counts: Optional[ConfidenceClassCounts]
) -> ConfidencePredictorHead:
    num_chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    batch_norm = bool(args.batch_norm)
    confidence_loss = ConfidenceLoss(
        args.confidence_weight, args.focusing_gamma)
    return ConfidencePredictorHead(
        head_name, num_chunk_frames, num_classes, confidence_loss,
        args.layer_weight_decay, batch_norm, args.dropout, args.width,
        args.head_layers, confidence_class_counts)


def _confidence_aux_predictor_head(
        args: SharedArgs, head_name: str, num_classes: int,
        radii: List[float]) -> ConfidenceAuxPredictorHead:
    num_chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    batch_norm = bool(args.batch_norm)
    confidence_aux_loss = ConfidenceAuxLoss(
        args.confidence_aux_weight, args.focusing_gamma, len(radii))
    return ConfidenceAuxPredictorHead(
        head_name, radii, num_chunk_frames, num_classes, confidence_aux_loss,
        args.layer_weight_decay, batch_norm, args.dropout, args.width,
        args.head_layers)


def _segmentation_trainer_head(
        args: SharedArgs, head_name: str, training_set: Dataset,
        class_weights: np.ndarray) -> SegmentationTrainerHead:
    if args.segmentation_weight_temperature:
        weight_creator = SegmentationWeightCreator(
            training_set, args.segmentation_weight_temperature, class_weights)
    else:
        weight_creator = IdentityWeightCreator(class_weights)
    segmentation_predictor_head = _segmentation_predictor_head(
        args, head_name, training_set.num_classes_from_task[Task.SEGMENTATION])
    return SegmentationTrainerHead(segmentation_predictor_head, weight_creator)


def _segmentation_predictor_head(
        args: SharedArgs, head_name: str,
        num_classes: int) -> SegmentationPredictorHead:
    num_chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    batch_norm = bool(args.batch_norm)
    segmentation_loss = SegmentationLoss(
        args.segmentation_weight, args.focusing_gamma)
    return SegmentationPredictorHead(
        head_name, num_chunk_frames, num_classes, segmentation_loss,
        args.layer_weight_decay, batch_norm, args.dropout, args.width,
        args.head_layers)


def _video_chunk_iterator_provider(
        args: SharedArgs, chunk_prediction_border: float
) -> VideoChunkIteratorProvider:
    num_chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    num_border_frames = _num_border_frames(args, chunk_prediction_border)
    return VideoChunkIteratorProvider(num_chunk_frames, num_border_frames)


def _create_keras_model(
        args: SharedArgs, input_shape: InputShape,
        predictor_heads: List[PredictorHeadInterface]) -> Model:
    main_input = create_main_input(input_shape)
    if args.input_weight_decay is None:
        input_weight_decay = args.layer_weight_decay
    else:
        input_weight_decay = args.input_weight_decay
    num_reduction_out_channels = args.reduction_width_factor * args.width
    num_chunk_frames = input_shape[0]
    num_features = input_shape[1]
    input_reduction_out = input_reduction(
        main_input, num_features, num_reduction_out_channels,
        input_weight_decay, args.batch_norm, args.dropout)
    nodes = _create_backbone(
        args, num_chunk_frames, num_reduction_out_channels, input_reduction_out)
    output_tensors = _create_output_tensors(predictor_heads, nodes)
    model = Model(main_input, output_tensors)
    maybe_convert_model_to_sam(model, args.sam_rho, SAM_EPSILON)
    return model


def _create_backbone(
        args: SharedArgs, num_chunk_frames: int,
        num_reduction_out_channels: int, input_reduction_out: Tensor) -> Nodes:
    if args.backbone == BACKBONE_UNET:
        return _unet_backbone(
            args, input_reduction_out, num_reduction_out_channels)
    elif args.backbone == BACKBONE_TRANSFORMER_ENCODER:
        return _transformer_encoder_backbone(
            args, input_reduction_out, num_reduction_out_channels,
            num_chunk_frames)
    else:
        raise ValueError(f"Unrecognized value for backbone: {args.backbone}")


def _transformer_encoder_backbone(
        args: SharedArgs, input_reduction_out: Tensor,
        num_reduction_out_channels: int, num_chunk_frames: int) -> Nodes:
    if args.transformer_hidden_groups == 0:
        hidden_groups = args.transformer_hidden_layers
    else:
        hidden_groups = args.transformer_hidden_groups
    transformer_encoder_config = TransformerEncoderConfig(
        embedding_size=num_reduction_out_channels,
        num_chunk_frames=num_chunk_frames,
        initializer_range=TRANSFORMER_INITIALIZER_RANGE,
        layer_dropout=TRANSFORMER_LAYER_DROPOUT,
        layer_norm_eps=TRANSFORMER_LAYER_NORM_EPS,
        do_stable_layer_norm=TRANSFORMER_STABLE_LAYER_NORM,
        hidden_layers=args.transformer_hidden_layers,
        hidden_groups=hidden_groups,
        hidden_size=args.transformer_hidden_size,
        hidden_dropout=TRANSFORMER_HIDDEN_DROPOUT,
        hidden_activation=TRANSFORMER_HIDDEN_ACTIVATION,
        attention_heads=args.transformer_attention_heads,
        attention_dropout=TRANSFORMER_ATTENTION_DROPOUT,
        intermediate_size=args.transformer_intermediate_size,
        activation_dropout=TRANSFORMER_ACTIVATION_DROPOUT,
        positional_embedding=TRANSFORMER_POSITIONAL_EMBEDDING,
        convolutional_positional_embedding_kernel=TRANSFORMER_CONVOLUTIONAL_POSITIONAL_EMBEDDING_KERNEL,
        convolutional_positional_embedding_groups=TRANSFORMER_CONVOLUTIONAL_POSITIONAL_EMBEDDING_GROUPS,
        convolutional_positional_embedding_activation=TRANSFORMER_CONVOLUTIONAL_POSITION_EMBEDDING_ACTIVATION,
        name=TRANSFORMER_ENCODER_NAME)
    return create_transformer_encoder_backbone(
        transformer_encoder_config, input_reduction_out)


def _unet_backbone(
        args: SharedArgs, input_reduction_out: Tensor,
        num_reduction_out_channels: int) -> Nodes:
    unet_weight_decay = args.layer_weight_decay
    bottom_up_layers = _unet_bottom_up_layers(
        args, input_reduction_out, num_reduction_out_channels,
        unet_weight_decay)
    top_down_stack = _unet_top_down_stack(args, unet_weight_decay)
    return create_unet_backbone(
        bottom_up_layers, args.unet_layers_start, args.unet_layers_end,
        top_down_stack)


def _unet_top_down_stack(
        args: SharedArgs,
        unet_weight_decay: float) -> TopDownStackInterface:
    pre_activated = UNET_RESIDUAL_V2
    if args.unet_combiner == UNET_COMBINER_CONCATENATION:
        combiner = ConcatenationCombiner()
    elif args.unet_combiner == UNET_COMBINER_ADDITION:
        combiner = AdditionCombiner(
            UNET_ADDITION_COMBINER_PRE_CONVOLVE, unet_weight_decay,
            args.batch_norm, args.dropout, UNET_ACTIVATION, pre_activated)
    elif args.unet_combiner == UNET_COMBINER_IGNORE:
        combiner = IgnoreCombiner()
    else:
        raise ValueError(f"Unknown unetcombiner: {args.unet_combiner}")
    if args.unet_upsampler == UNET_UPSAMPLER_TRANSPOSE:
        # Tried 2, 3 and 4 here, and they seem to work similarly when used
        # in combination with later convolutions. I can leave it at 2,
        # since that is more minimalistic. Original u-net paper used 2,
        # but some other implementations I found online used 3 or 4.
        conv_transpose_kernel_size = (2, 1)
        upsampler = ConvTransposeUpsampler(
            conv_transpose_kernel_size, "same", unet_weight_decay,
            args.batch_norm, args.dropout, UNET_ACTIVATION, pre_activated)
    elif args.unet_upsampler == UNET_UPSAMPLER_UPSAMPLING:
        # Tried 2, 3, and 4 here and maybe 2 worked best, but not enough
        # experiments to confirm.
        upsampling_conv_kernel_size = (2, 1)
        upsampler = UpsamplingUpsampler(
            UNET_UPSAMPLING_INTERPOLATION, UNET_UPSAMPLING_CONVOLVE,
            upsampling_conv_kernel_size, "same", unet_weight_decay,
            args.batch_norm, args.dropout, UNET_ACTIVATION,
            pre_activated)
    else:
        raise ValueError(f"Unknown unetupsampler: {args.unet_upsampler}")
    strided_block = _strided_block(args, UNET_RESIDUAL_V2, unet_weight_decay)
    convolution_stack = BasicConvolutionStack(
        strided_block, UNET_TOP_DOWN_STACK_NUM_BLOCKS)
    return BasicTopDownStack(
        upsampler, combiner, convolution_stack, pre_activated)


def _unet_bottom_up_layers(
        args: SharedArgs, input_reduction_out: Tensor,
        base_num_filters: int, unet_weight_decay: float) -> List[BottomUpLayer]:
    strided_block = _strided_block(args, UNET_RESIDUAL_V2, unet_weight_decay)
    if UNET_DOWNSAMPLE == UNET_DOWNSAMPLE_STRIDE:
        bottom_up_stack = StridedBottomUpStack(
            strided_block, UNET_BOTTOM_UP_STACK_NUM_BLOCKS, False)
    elif (UNET_DOWNSAMPLE == UNET_DOWNSAMPLE_POOLING_MAX or
          UNET_DOWNSAMPLE == UNET_DOWNSAMPLE_POOLING_AVERAGE):
        bottom_up_convolution_stack = BasicConvolutionStack(
            strided_block, UNET_BOTTOM_UP_STACK_NUM_BLOCKS)
        bottom_up_stack = PoolingBottomUpStack(
            UNET_DOWNSAMPLE, bottom_up_convolution_stack)
    else:
        raise ValueError(f"UNET_DOWNSAMPLE not recognized: {UNET_DOWNSAMPLE}")
    return create_bottom_up_layers(
        input_reduction_out, args.unet_layers_end, base_num_filters,
        UNET_MAX_FILTERS, bottom_up_stack)


def _strided_block(
        args: SharedArgs, unet_residual_v2: bool,
        unet_weight_decay: float) -> StridedBlockInterface:
    if unet_residual_v2:
        return ResidualBlockV2(
            UNET_BLOCK_NUM_CONVOLUTIONS, bool(args.skip_init),
            UNET_BLOCK_BOTTLENECK, unet_weight_decay, args.batch_norm,
            args.dropout)
    else:
        return ResidualBlock(
            UNET_BLOCK_NUM_CONVOLUTIONS, UNET_BLOCK_BOTTLENECK,
            unet_weight_decay, args.batch_norm, args.dropout)


def _num_border_frames(
        args: SharedArgs, chunk_prediction_border: float) -> int:
    num_chunk_frames = math.floor(args.frame_rate * args.chunk_duration)
    if args.profile:
        # Make the number of border frames be as large as possible, so that
        # prediction for a video requires many chunks.
        return (num_chunk_frames - 1) // 2
    # For a given chunk, we require accepting at least
    # MIN_ACCEPTED_PREDICTION_SECONDS from the chunk.
    min_num_accepted_prediction_frames = math.floor(
        args.frame_rate * MIN_ACCEPTED_PREDICTION_SECONDS)
    max_num_border_frames = (
            (num_chunk_frames - min_num_accepted_prediction_frames) // 2)
    desired_num_border_frames = math.ceil(
        args.frame_rate * chunk_prediction_border)
    return min(desired_num_border_frames, max_num_border_frames)


def _create_output_tensors(
        predictor_heads: List[PredictorHeadInterface],
        nodes: Nodes) -> List[Tensor]:
    return [
        predictor_head.create_tensor(nodes)
        for predictor_head in predictor_heads]


def _load_model(args: SharedArgs, model_path: str) -> Model:
    # This loads a compiled models (assuming the models was compiled
    # originally).
    model = load_model(model_path)
    maybe_convert_model_to_sam(model, args.sam_rho, SAM_EPSILON)
    return model


def _update_keras_custom_objects(heads: List[PredictorHeadInterface]) -> None:
    # Add the loss functions to custom_objects so that Keras can access them
    # when loading the models from disk.
    losses_dict = {head.loss_name: head.loss for head in heads}
    get_custom_objects().update(losses_dict)
    get_custom_objects().update({"SAMModel": SAMModel})
