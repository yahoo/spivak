# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
import os
import pickle
from pathlib import Path
from typing import Optional

from spivak.data.dataset_splits import SPLIT_KEY_TEST, ALL_SPLIT_KEYS

DATASET_TYPE_CUSTOM_SPOTTING = "custom_spotting"
DATASET_TYPE_CUSTOM_SEGMENTATION = "custom_segmentation"
DATASET_TYPE_CUSTOM_SPOTTING_AND_SEGMENTATION = \
    "custom_spotting_and_segmentation"
DATASET_TYPE_SOCCERNET = "soccernet"
DATASET_TYPE_SOCCERNET_V2 = "soccernet_v2"
DATASET_TYPE_SOCCERNET_V2_CAMERA_SEGMENTATION = \
    "soccernet_v2_camera_segmentation"
DATASET_TYPE_SOCCERNET_V2_SPOTTING_AND_CAMERA_SEGMENTATION = \
    "soccernet_v2_spotting_and_camera_segmentation"
DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION = \
    "soccernet_v2_challenge_validation"
DATASET_TYPE_SOCCERNET_V2_CHALLENGE = "soccernet_v2_challenge"
SHARED_ARGS_FILE = "shared_args.pkl"
LEARNING_RATE_DECAY_EXPONENTIAL = "exponential"
LEARNING_RATE_DECAY_LEGACY = "legacy"
LEARNING_RATE_DECAY_LINEAR = "linear"
LEARNING_RATE_DECAY_NONE = "none"
OPTIMIZER_SGD = "sgd"
OPTIMIZER_ADAM = "adam"
DETECTOR_DENSE = "dense"
DETECTOR_DENSE_DELTA = "dense_delta"
DETECTOR_AVERAGING_CONFIDENCE = "averaging_confidence"
DETECTOR_AVERAGING_DELTA = "averaging_delta"
SAMPLING_UNIFORM = "uniform"
SAMPLING_WEIGHTED = "weighted"
BACKBONE_UNET = "unet"
BACKBONE_TRANSFORMER_ENCODER = "transformer_encoder"
UNET_COMBINER_CONCATENATION = "concatenation"
UNET_COMBINER_ADDITION = "addition"
UNET_COMBINER_IGNORE = "ignore"
UNET_UPSAMPLER_TRANSPOSE = "transpose"
UNET_UPSAMPLER_UPSAMPLING = "upsampling"
NMS_DECAY_SUPPRESS = "suppress"
NMS_DECAY_LINEAR = "linear"
NMS_DECAY_GAUSSIAN = "gaussian"
DATASET_TYPE_CHOICES = [
    DATASET_TYPE_SOCCERNET,
    DATASET_TYPE_SOCCERNET_V2,
    DATASET_TYPE_SOCCERNET_V2_CAMERA_SEGMENTATION,
    DATASET_TYPE_SOCCERNET_V2_SPOTTING_AND_CAMERA_SEGMENTATION,
    DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION,
    DATASET_TYPE_SOCCERNET_V2_CHALLENGE,
    DATASET_TYPE_CUSTOM_SPOTTING,
    DATASET_TYPE_CUSTOM_SEGMENTATION,
    DATASET_TYPE_CUSTOM_SPOTTING_AND_SEGMENTATION]


class Defaults:
    """
    We define all the defaults in this class, which in effect just acts
    like a namespace. The defaults are used by both the argument parser and
    SharedArgs (defined below). SharedArgs defines the set of input arguments
    for use as a library (as opposed to command-line use with the argument
    parser).
    """
    # Inputs
    RESULTS_DIR: Optional[str] = None
    PRETRAINED_PATH: Optional[str] = None
    FEATURES_DIR: Optional[str] = None
    LABELS_DIR: Optional[str] = None
    MODEL_WEIGHTS: Optional[str] = None
    MODEL: Optional[str] = None
    CONFIG_DIR: Optional[str] = None
    SPLITS_DIR: Optional[str] = None
    DATASET_TYPE = DATASET_TYPE_SOCCERNET
    FEATURE_NAME = "ResNET_PCA512"
    FRAME_RATE = 2.0
    # Training
    EPOCHS = 1000
    VALIDATION_EPOCHS = 10
    SAVE_EPOCHS = 10
    BATCH_SIZE = 256
    MIXUP_ALPHA = 0.0
    SAMPLING = SAMPLING_UNIFORM
    # This fraction is used when sampling chunks (not frames). It's probably
    # good to avoid setting this too low, so that we get more variety in the
    # training data (at least some sampling from negatives), though
    # empirically low values work better.
    SAMPLING_NEGATIVE_FRACTION = 0.1
    # Experiments showed it's better to use most or all of the data for
    # training the confidence and delta, so we leave these at 1.0 by default.
    # The results for the confidence might have been slightly better at 0.5,
    # but it simplifies things to just set it to 1.0.
    SAMPLING_NEGATIVE_RATE_DELTA = 1.0
    SAMPLING_NEGATIVE_RATE_CONFIDENCE = 1.0
    POSITIVE_WEIGHT_CONFIDENCE = 0.03
    POSITIVE_WEIGHT_DELTA = 1.0
    SEGMENTATION_WEIGHT_TEMPERATURE = 0.0
    CACHE_FEATURES = 1
    OPTIMIZER = OPTIMIZER_ADAM
    # I wasn't able to get speed improvements when using mixed precision with
    # my models (speed became slightly worse).
    MIXED_PRECISION = 0
    SAM_RHO = 0.0
    # For Adam with linear learning rate decay on SoccerNet, set LEARNING_RATE
    # to 2e-3. For SGD with yolo detection layer and linear learning rate decay
    # on SoccerNet, set LEARNING_RATE to 4e-4.
    LEARNING_RATE = 2e-3
    LEARNING_RATE_DECAY = LEARNING_RATE_DECAY_LINEAR
    LEARNING_RATE_DECAY_EXPONENTIAL_RATE = 0.1
    LEARNING_RATE_DECAY_EXPONENTIAL_EPOCHS = 300
    LEARNING_RATE_DECAY_LINEAR_EPOCHS = 1000
    DELTA_WEIGHT = 0.0
    DELTA_AUX_WEIGHT = 0.0
    CONFIDENCE_WEIGHT = 0.0
    # Experiments show that we don't need an auxiliary confidence output
    # if we already have the auxiliary delta output.
    CONFIDENCE_AUX_WEIGHT = 0.0
    SEGMENTATION_WEIGHT = 0.0
    # Architecture
    DETECTOR = DETECTOR_DENSE
    BACKBONE = BACKBONE_UNET
    # SkipInit didn't help (and maybe hurt slightly), even when using many
    # layers in a u-net. However, it turned out to be very important when
    # removing the u-net bottom-up combination paths (-uc ignore), as the
    # optimization seems to get much more difficult in that case.
    SKIP_INIT = 0
    # From early initial experiments, concatenation seems to do just as well as
    # addition without the pre-convolution. Addition seems simpler,
    # so setting that as the default.
    UNET_COMBINER = UNET_COMBINER_ADDITION
    # From some limited experiments, transpose convolution seems to do better
    # than upsampling with nearest neighbors.
    UNET_UPSAMPLER = UNET_UPSAMPLER_TRANSPOSE
    # Experiments show that using 0 and 6 below works well.
    UNET_LAYERS_START = 0
    UNET_LAYERS_END = 6
    # These default settings follow Baidu's technical report.
    TRANSFORMER_HIDDEN_LAYERS = 3
    # A value of 0 for hidden groups indicates that there should be no
    # groups, ie each layer is different, which can also be seen as having
    # the number of groups being equal to the number of layers.
    TRANSFORMER_HIDDEN_GROUPS = 0
    TRANSFORMER_HIDDEN_SIZE = 64
    # A rule of thumb that people use os to set
    # TRANSFORMER_INTERMEDIATE_SIZE to 4 * TRANSFORMER_HIDDEN_SIZE.
    TRANSFORMER_INTERMEDIATE_SIZE = 256
    TRANSFORMER_ATTENTION_HEADS = 4
    # For reference, here is what the HUBERT base setup looks like.
    # TRANSFORMER_HIDDEN_LAYERS = 12
    # TRANSFORMER_HIDDEN_SIZE = 768
    # TRANSFORMER_INTERMEDIATE_SIZE = 3072
    # TRANSFORMER_ATTENTION_HEADS = 8
    # And here is the HUBERT large setup:
    # TRANSFORMER_HIDDEN_LAYERS = 24
    # TRANSFORMER_HIDDEN_SIZE = 1024
    # TRANSFORMER_INTERMEDIATE_SIZE = 4096
    # TRANSFORMER_ATTENTION_HEADS = 16
    # CHUNK_DURATION and MIN_VALID_CHUNK_DURATION are measured in seconds.
    # 112.0 seconds at 2 frames per second gives a chunk with 224 elements,
    # which our u-net with 6 layers is able to handle. In that case, the last
    # layer has 7 elements, which is 224 / 2**5.
    CHUNK_DURATION = 112.0
    MIN_VALID_CHUNK_DURATION = 112.0
    # NMS_WINDOW is measured in seconds.
    NMS_WINDOW = 20.0
    # Measured in seconds. Making this large in order to only take the
    # central portion of the inference tends to improve results slightly.
    CHUNK_PREDICTION_BORDER = 50.0
    # So that validation runs in a reasonable amount of time, we reduce the
    # border size specifically just for validation.
    VALIDATION_CHUNK_PREDICTION_BORDER = 20.0
    WIDTH = 16
    REDUCTION_WIDTH_FACTOR = 4
    # Some experiments showed there was no benefit in adding more layers to
    # the heads, so we kept this at 1.
    HEAD_LAYERS = 1
    DECOUPLED_WEIGHT_DECAY = 0.0
    LAYER_WEIGHT_DECAY = 0.0
    # None indicates the same weight decay should be used on the input layers
    # as that is used on the rest of the layers.
    INPUT_WEIGHT_DECAY = None
    BATCH_NORM = 0
    DROPOUT = 0.0
    # Evaluation
    EVALUATE = 1
    CREATE_CONFUSION = 1
    APPLY_NMS = 1
    NMS_DECAY = NMS_DECAY_SUPPRESS
    NMS_DECAY_LINEAR_MIN = 0.0
    TEST_SPLIT = SPLIT_KEY_TEST
    PRUNE_CLASSES = 0
    SEGMENTATION_EVALUATION_OLD = 0
    TEST_PREDICT = 1
    TEST_NMS_ONLY = 0
    TEST_SAVE_LABELS = 0
    TEST_SAVE_SPOTTING_JSONS = 0
    PROFILE = 0
    # Confidence loss
    # Performance seems to be a bit better without this initialization, so
    # better to just use the default initialization with zeros instead.
    INITIALIZE_BIASES = 0
    # The original focal loss paper suggests 2.0 is a good value for gamma. For
    # them, values between 0.5 and 5.0 worked well.
    FOCUSING_GAMMA = 0.0
    # Radius used in dense detection loss, specified in seconds. The 3.0 value
    # came from a series of experiments.
    DENSE_DETECTION_RADIUS = 3.0
    THROW_OUT_DELTA = 1
    DELTA_RADIUS_MULTIPLIER = 2.0
    # Non-cyclic delta is simpler and seems to work just as well, as long as
    # the range is restricted to being near actions.
    CYCLIC_DELTA = 0
    # The HUBER_DELTA specifies the point at which the Huber loss transitions
    # from behaving like an L2-loss to behaving like an L1-loss. The
    # differences used as input are between 0.0 and 1.0. The performance
    # doesn't seem to be very sensitive to this value. 0.5 works well.
    HUBER_DELTA = 0.5


class SharedArgs:

    def __init__(self) -> None:
        """The idea of this class is to provide an interface for being able
        to invoke all the functionality directly from code, in addition to being
        able to call it from the command line. Below, we initialize the values
        with the defaults defined above."""
        # inputs
        self.results_dir = Defaults.RESULTS_DIR
        self.pretrained_path = Defaults.PRETRAINED_PATH
        self.features_dir = Defaults.FEATURES_DIR
        self.labels_dir = Defaults.LABELS_DIR
        self.splits_dir = Defaults.SPLITS_DIR
        self.dataset_type = Defaults.DATASET_TYPE
        self.feature_name = Defaults.FEATURE_NAME
        self.frame_rate = Defaults.FRAME_RATE
        self.model_weights = Defaults.MODEL_WEIGHTS
        self.model = Defaults.MODEL
        self.config_dir = Defaults.CONFIG_DIR
        # training
        self.epochs = Defaults.EPOCHS
        self.validation_epochs = Defaults.VALIDATION_EPOCHS
        self.save_epochs = Defaults.SAVE_EPOCHS
        self.batch_size = Defaults.BATCH_SIZE
        self.mixup_alpha = Defaults.MIXUP_ALPHA
        self.sampling = Defaults.SAMPLING
        self.sampling_negative_fraction = Defaults.SAMPLING_NEGATIVE_FRACTION
        self.sampling_negative_rate_delta = \
            Defaults.SAMPLING_NEGATIVE_RATE_DELTA
        self.sampling_negative_rate_confidence = \
            Defaults.SAMPLING_NEGATIVE_RATE_CONFIDENCE
        self.positive_weight_confidence = Defaults.POSITIVE_WEIGHT_CONFIDENCE
        self.positive_weight_delta = Defaults.POSITIVE_WEIGHT_DELTA
        self.segmentation_weight_temperature = \
            Defaults.SEGMENTATION_WEIGHT_TEMPERATURE
        self.cache_features = Defaults.CACHE_FEATURES
        self.optimizer = Defaults.OPTIMIZER
        self.mixed_precision = Defaults.MIXED_PRECISION
        self.sam_rho = Defaults.SAM_RHO
        self.learning_rate = Defaults.LEARNING_RATE
        self.learning_rate_decay = Defaults.LEARNING_RATE_DECAY
        self.learning_rate_decay_exponential_epochs = \
            Defaults.LEARNING_RATE_DECAY_EXPONENTIAL_EPOCHS
        self.learning_rate_decay_exponential_rate = \
            Defaults.LEARNING_RATE_DECAY_EXPONENTIAL_RATE
        self.learning_rate_decay_linear_epochs = \
            Defaults.LEARNING_RATE_DECAY_LINEAR_EPOCHS
        self.delta_weight = Defaults.DELTA_WEIGHT
        self.delta_aux_weight = Defaults.DELTA_AUX_WEIGHT
        self.confidence_weight = Defaults.CONFIDENCE_WEIGHT
        self.confidence_aux_weight = Defaults.CONFIDENCE_AUX_WEIGHT
        self.segmentation_weight = Defaults.SEGMENTATION_WEIGHT
        # Architecture
        self.detector = Defaults.DETECTOR
        self.backbone = Defaults.BACKBONE
        self.skip_init = Defaults.SKIP_INIT
        self.unet_combiner = Defaults.UNET_COMBINER
        self.unet_upsampler = Defaults.UNET_UPSAMPLER
        self.unet_layers_start = Defaults.UNET_LAYERS_START
        self.unet_layers_end = Defaults.UNET_LAYERS_END
        self.transformer_hidden_layers = Defaults.TRANSFORMER_HIDDEN_LAYERS
        self.transformer_hidden_groups = Defaults.TRANSFORMER_HIDDEN_GROUPS
        self.transformer_hidden_size = Defaults.TRANSFORMER_HIDDEN_SIZE
        self.transformer_intermediate_size = \
            Defaults.TRANSFORMER_INTERMEDIATE_SIZE
        self.transformer_attention_heads = Defaults.TRANSFORMER_ATTENTION_HEADS
        self.chunk_duration = Defaults.CHUNK_DURATION
        self.min_valid_chunk_duration = Defaults.MIN_VALID_CHUNK_DURATION
        self.nms_window = Defaults.NMS_WINDOW
        self.chunk_prediction_border = Defaults.CHUNK_PREDICTION_BORDER
        self.validation_chunk_prediction_border = \
            Defaults.VALIDATION_CHUNK_PREDICTION_BORDER
        self.width = Defaults.WIDTH
        self.reduction_width_factor = Defaults.REDUCTION_WIDTH_FACTOR
        self.head_layers = Defaults.HEAD_LAYERS
        self.decoupled_weight_decay = Defaults.DECOUPLED_WEIGHT_DECAY
        self.layer_weight_decay = Defaults.LAYER_WEIGHT_DECAY
        self.input_weight_decay = Defaults.INPUT_WEIGHT_DECAY
        self.batch_norm = Defaults.BATCH_NORM
        self.dropout = Defaults.DROPOUT
        # Evaluation
        self.evaluate = Defaults.EVALUATE
        self.create_confusion = Defaults.CREATE_CONFUSION
        self.apply_nms = Defaults.APPLY_NMS
        self.nms_decay = Defaults.NMS_DECAY
        self.nms_decay_linear_min = Defaults.NMS_DECAY_LINEAR_MIN
        self.test_split = Defaults.TEST_SPLIT
        self.prune_classes = Defaults.PRUNE_CLASSES
        self.segmentation_evaluation_old = Defaults.SEGMENTATION_EVALUATION_OLD
        self.test_predict = Defaults.TEST_PREDICT
        self.test_nms_only = Defaults.TEST_NMS_ONLY
        self.test_save_labels = Defaults.TEST_SAVE_LABELS
        self.test_save_spotting_jsons = Defaults.TEST_SAVE_SPOTTING_JSONS
        self.profile = Defaults.PROFILE
        # Confidence loss
        self.initialize_biases = Defaults.INITIALIZE_BIASES
        self.focusing_gamma = Defaults.FOCUSING_GAMMA
        # Dense detection loss in seconds
        self.dense_detection_radius = Defaults.DENSE_DETECTION_RADIUS
        self.throw_out_delta = Defaults.THROW_OUT_DELTA
        self.delta_radius_multiplier = Defaults.DELTA_RADIUS_MULTIPLIER
        self.cyclic_delta = Defaults.CYCLIC_DELTA
        self.huber_delta = Defaults.HUBER_DELTA

    @staticmethod
    def from_parsed_args(args: argparse.Namespace) -> "SharedArgs":
        shared_args = SharedArgs()
        # inputs
        shared_args.results_dir = args.results_dir
        shared_args.pretrained_path = args.pretrained_path
        shared_args.features_dir = args.features_dir
        shared_args.labels_dir = args.labels_dir
        shared_args.splits_dir = args.splits_dir
        shared_args.dataset_type = args.dataset_type
        shared_args.feature_name = args.feature_name
        shared_args.frame_rate = args.frame_rate
        shared_args.model_weights = args.model_weights
        shared_args.model = args.model
        shared_args.config_dir = args.config_dir
        # training
        shared_args.epochs = args.epochs
        shared_args.validation_epochs = args.validation_epochs
        shared_args.save_epochs = args.save_epochs
        shared_args.batch_size = args.batch_size
        shared_args.mixup_alpha = args.mixup_alpha
        shared_args.sampling = args.sampling
        shared_args.sampling_negative_fraction = \
            args.sampling_negative_fraction
        shared_args.sampling_negative_rate_delta = \
            args.sampling_negative_rate_delta
        shared_args.sampling_negative_rate_confidence = \
            args.sampling_negative_rate_confidence
        shared_args.cache_features = args.cache_features
        shared_args.positive_weight_confidence = \
            args.positive_weight_confidence
        shared_args.positive_weight_delta = args.positive_weight_delta
        shared_args.segmentation_weight_temperature = \
            args.segmentation_weight_temperature
        shared_args.optimizer = args.optimizer
        shared_args.mixed_precision = args.mixed_precision
        shared_args.sam_rho = args.sam_rho
        shared_args.learning_rate = args.learning_rate
        shared_args.learning_rate_decay = args.learning_rate_decay
        shared_args.learning_rate_decay_exponential_rate = \
            args.learning_rate_decay_exponential_rate
        shared_args.learning_rate_decay_exponential_epochs = \
            args.learning_rate_decay_exponential_epochs
        shared_args.learning_rate_decay_linear_epochs = \
            args.learning_rate_decay_linear_epochs
        shared_args.delta_weight = args.delta_weight
        shared_args.delta_aux_weight = args.delta_aux_weight
        shared_args.confidence_weight = args.confidence_weight
        shared_args.confidence_aux_weight = \
            args.confidence_aux_weight
        shared_args.segmentation_weight = \
            args.segmentation_weight
        # Architecture
        shared_args.detector = args.detector
        shared_args.backbone = args.backbone
        shared_args.skip_init = args.skip_init
        shared_args.unet_combiner = args.unet_combiner
        shared_args.unet_upsampler = args.unet_upsampler
        shared_args.unet_layers_start = args.unet_layers_start
        shared_args.unet_layers_end = args.unet_layers_end
        shared_args.transformer_hidden_layers = \
            args.transformer_hidden_layers
        shared_args.transformer_hidden_groups = \
            args.transformer_hidden_groups
        shared_args.transformer_hidden_size = args.transformer_hidden_size
        shared_args.transformer_intermediate_size = \
            args.transformer_intermediate_size
        shared_args.transformer_attention_heads = \
            args.transformer_attention_heads
        shared_args.chunk_duration = args.chunk_duration
        shared_args.min_valid_chunk_duration = args.min_valid_chunk_duration
        shared_args.nms_window = args.nms_window
        shared_args.chunk_prediction_border = args.chunk_prediction_border
        shared_args.validation_chunk_prediction_border = \
            args.validation_chunk_prediction_border
        shared_args.width = args.width
        shared_args.reduction_width_factor = args.reduction_width_factor
        shared_args.head_layers = args.head_layers
        shared_args.decoupled_weight_decay = args.decoupled_weight_decay
        shared_args.layer_weight_decay = args.layer_weight_decay
        shared_args.input_weight_decay = args.input_weight_decay
        shared_args.batch_norm = args.batch_norm
        shared_args.dropout = args.dropout
        # Evaluation
        shared_args.evaluate = args.evaluate
        shared_args.create_confusion = args.create_confusion
        shared_args.apply_nms = args.apply_nms
        shared_args.nms_decay = args.nms_decay
        shared_args.nms_decay_linear_min = args.nms_decay_linear_min
        shared_args.test_split = args.test_split
        shared_args.prune_classes = args.prune_classes
        shared_args.segmentation_evaluation_old = \
            args.segmentation_evaluation_old
        shared_args.test_predict = args.test_predict
        shared_args.test_nms_only = args.test_nms_only
        shared_args.test_save_labels = args.test_save_labels
        shared_args.test_save_spotting_jsons = args.test_save_spotting_jsons
        shared_args.profile = args.profile
        # Confidence loss
        shared_args.initialize_biases = args.initialize_biases
        shared_args.focusing_gamma = args.focusing_gamma
        # Dense detection loss in seconds
        shared_args.dense_detection_radius = args.dense_detection_radius
        shared_args.throw_out_delta = args.throw_out_delta
        shared_args.delta_radius_multiplier = args.delta_radius_multiplier
        shared_args.cyclic_delta = args.cyclic_delta
        shared_args.huber_delta = args.huber_delta
        return shared_args

    def save(self, save_dir: str) -> None:
        args_path = os.path.join(save_dir, SHARED_ARGS_FILE)
        with open(args_path, "wb") as pickle_file:
            return pickle.dump(self, pickle_file)

    @staticmethod
    def load(save_dir: str) -> "SharedArgs":
        args_path = os.path.join(save_dir, SHARED_ARGS_FILE)
        with open(args_path, "rb") as pickle_file:
            return pickle.load(pickle_file)


def get_args() -> SharedArgs:
    parser = _create_parser()
    return SharedArgs.from_parsed_args(parser.parse_args())


def dir_str_to_path(dir_str: Optional[str]) -> Path:
    if not dir_str:
        raise ValueError(f"Missing input directory")
    dir_path = Path(dir_str)
    if not dir_path.is_dir():
        raise ValueError(f"Not a valid input directory: {dir_path}")
    return dir_path


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", "-rd", help="Path to save/read the inference results",
        type=str, required=False, default=Defaults.RESULTS_DIR)
    parser.add_argument(
        "--pretrained_path", "-pp", help=(
            "Pretrained models for initializing weights"),
        type=str, required=False, default=Defaults.PRETRAINED_PATH)
    parser.add_argument(
        "--features_dir", "-fd", help="Directory containing feature files.",
        type=str, required=False, default=Defaults.FEATURES_DIR)
    parser.add_argument(
        "--labels_dir", "-ld", help="Directory containing label files.",
        type=str, required=False, default=Defaults.LABELS_DIR)
    parser.add_argument(
        "--splits_dir", "-sd", help="Directory containing split files.",
        type=str, required=False, default=Defaults.SPLITS_DIR)
    parser.add_argument(
        "--dataset_type", "-dt", help="Dataset type", type=str, required=False,
        default=Defaults.DATASET_TYPE, choices=DATASET_TYPE_CHOICES)
    parser.add_argument(
        "--feature_name", "-fn", help="String describing the feature", type=str,
        default=Defaults.FEATURE_NAME)
    parser.add_argument(
        "--frame_rate", "-fr", help="Frame-rate of the features", type=float,
        default=Defaults.FRAME_RATE)
    parser.add_argument(
        "--model_weights", "-mw", help="Model weights path", type=str,
        default=Defaults.MODEL_WEIGHTS)
    parser.add_argument(
        "--model", "-m", help="Model path", type=str, default=Defaults.MODEL)
    parser.add_argument(
        "--config_dir", "-cd", help=(
            "Directory containing a set of config files"), type=str,
        required=True)
    parser.add_argument(
        "--epochs", "-e", help="Number of training epochs", type=int,
        default=Defaults.EPOCHS)
    parser.add_argument(
        "--validation_epochs", "-ve", help="How frequently to run validation",
        type=int, default=Defaults.VALIDATION_EPOCHS)
    parser.add_argument(
        "--save_epochs", "-se", help="How frequently to save the model to disk",
        type=int, default=Defaults.SAVE_EPOCHS)
    parser.add_argument(
        "--batch_size", "-bs", help="Batch size (in number of chunks)",
        type=int, default=Defaults.BATCH_SIZE)
    parser.add_argument(
        "--mixup_alpha", "-mu", help="Alpha used for mix-up", type=float,
        default=Defaults.MIXUP_ALPHA)
    parser.add_argument(
        "--sampling", "-sm",
        help="Sampling strategy for getting chunks from videos", type=str,
        default=Defaults.SAMPLING, choices=[
            SAMPLING_UNIFORM, SAMPLING_WEIGHTED])
    parser.add_argument(
        "--positive_weight_confidence", "-pwc",
        help="Overall weight of positive samples relative to negative "
             "samples, between 0.0 and 1.0. 0.0 (the default) indicates that "
             "the samples should not be weighted.", type=float,
        default=Defaults.POSITIVE_WEIGHT_CONFIDENCE)
    parser.add_argument(
        "--positive_weight_delta", "-pwdl",
        help="Overall weight of positive samples relative to negative "
             "samples, between 0.0 and 1.0. 0.0 (the default) indicates that "
             "the samples should not be weighted.", type=float,
        default=Defaults.POSITIVE_WEIGHT_DELTA)
    parser.add_argument(
        "--segmentation_weight_temperature", "-swt",
        help="Temperature of the segmentation class weights.", type=float,
        default=Defaults.SEGMENTATION_WEIGHT_TEMPERATURE)
    parser.add_argument(
        "--cache_features", "-cf",
        help="Whether to cache the features during training.", type=int,
        default=Defaults.CACHE_FEATURES, choices=[0, 1])
    parser.add_argument(
        "--sampling_negative_fraction", "-snf",
        help="Fraction of negative samples when sampling chunks from videos "
             "using the weighted sampling strategy",
        type=float, default=Defaults.SAMPLING_NEGATIVE_FRACTION)
    parser.add_argument(
        "--sampling_negative_rate_delta", "-snrd",
        help="Fraction of negative outputs to use from a chunk when training "
             "the delta head", type=float,
        default=Defaults.SAMPLING_NEGATIVE_RATE_DELTA)
    parser.add_argument(
        "--sampling_negative_rate_confidence", "-snrc",
        help="Fraction of negative outputs to use from a chunk when training "
             "the confidence head", type=float,
        default=Defaults.SAMPLING_NEGATIVE_RATE_CONFIDENCE)
    parser.add_argument(
        "--optimizer", "-z", help="Optimizer choice",
        type=str, default=Defaults.OPTIMIZER,
        choices=[OPTIMIZER_SGD, OPTIMIZER_ADAM])
    parser.add_argument(
        "--mixed_precision", "-mx",
        help="Whether to use mixed precision during training or not",
        type=int, default=Defaults.MIXED_PRECISION, choices=[0, 1])
    parser.add_argument(
        "--sam_rho", "-sr", help=(
            "Rho for Sharpness Aware Minimization (SAM). The paper suggests "
            "using 0.05."), type=float, default=Defaults.SAM_RHO,)
    parser.add_argument(
        "--learning_rate", "-lr", help="Learning rate", type=float,
        default=Defaults.LEARNING_RATE)
    parser.add_argument(
        "--learning_rate_decay", "-lrd",
        help="What type of learning rate decay to apply",
        type=str, default=Defaults.LEARNING_RATE_DECAY,
        choices=[LEARNING_RATE_DECAY_EXPONENTIAL, LEARNING_RATE_DECAY_LEGACY,
                 LEARNING_RATE_DECAY_LINEAR, LEARNING_RATE_DECAY_NONE])
    parser.add_argument(
        "--learning_rate_decay_exponential_rate", help=(
            "Learning rate multiplier applied when using exponential "
            "learning rate decay"), type=float,
        default=Defaults.LEARNING_RATE_DECAY_EXPONENTIAL_RATE)
    parser.add_argument(
        "--learning_rate_decay_exponential_epochs", help=(
            "Frequency in epochs at which to decay the learning rate when "
            "using exponential learning rate decay"), type=int,
        default=Defaults.LEARNING_RATE_DECAY_EXPONENTIAL_EPOCHS)
    parser.add_argument(
        "--learning_rate_decay_linear_epochs", "-lrdle", help=(
            "Frequency in epochs of the learning rate decay cycle when "
            "using linear learning rate decay"), type=int,
        default=Defaults.LEARNING_RATE_DECAY_LINEAR_EPOCHS)
    parser.add_argument(
        "--delta_weight", "-dw", help=(
            "Weight of delta loss term"), type=float,
        default=Defaults.DELTA_WEIGHT)
    parser.add_argument(
        "--delta_aux_weight", "-daw", help=(
            "Weight of auxiliary delta loss term"), type=float,
        default=Defaults.DELTA_AUX_WEIGHT)
    parser.add_argument(
        "--confidence_weight", "-cw", help=(
            "Weight of confidence loss term"), type=float,
        default=Defaults.CONFIDENCE_WEIGHT)
    parser.add_argument(
        "--confidence_aux_weight", "-caw", help=(
            "Weight of auxiliary confidence loss term"), type=float,
        default=Defaults.CONFIDENCE_AUX_WEIGHT)
    parser.add_argument(
        "--segmentation_weight", "-sw", help=(
            "Weight of the segmentation loss term"),
        type=float, default=Defaults.SEGMENTATION_WEIGHT)
    parser.add_argument(
        "--detector", "-dc", help="Detector module type", type=str,
        default=Defaults.DETECTOR,
        choices=[DETECTOR_DENSE, DETECTOR_DENSE_DELTA,
                 DETECTOR_AVERAGING_CONFIDENCE, DETECTOR_AVERAGING_DELTA])
    parser.add_argument(
        "--unet_combiner", "-uc",
        help="Unet method for combining bottom-up and top-down layers",
        type=str, default=Defaults.UNET_COMBINER,
        choices=[UNET_COMBINER_CONCATENATION, UNET_COMBINER_ADDITION,
                 UNET_COMBINER_IGNORE])
    parser.add_argument(
        "--unet_upsampler", "-uu",
        help="Unet method for upsampling in top-down layers", type=str,
        default=Defaults.UNET_UPSAMPLER,
        choices=[UNET_UPSAMPLER_TRANSPOSE, UNET_UPSAMPLER_UPSAMPLING])
    parser.add_argument(
        "--unet_layers_start", "-us", help=(
            "For the u-net, the first bottom-up layer that will be used by the "
            "top-down network"), type=int, default=Defaults.UNET_LAYERS_START)
    parser.add_argument(
        "--unet_layers_end", "-ue", help=(
            "For the u-net, the last bottom-up layer that will be used "
            "by the top-down network."), type=int,
        default=Defaults.UNET_LAYERS_END)
    parser.add_argument(
        "--transformer_hidden_layers", "-thl",
        help="Number of layers in the Transformer encoder.", type=int,
        default=Defaults.TRANSFORMER_HIDDEN_LAYERS)
    parser.add_argument(
        "--transformer_hidden_groups", "-thg",
        help="Number of layer groups in the Transformer encoder.", type=int,
        default=Defaults.TRANSFORMER_HIDDEN_GROUPS)
    parser.add_argument(
        "--transformer_hidden_size", "-ths",
        help="Size of the hidden Transformer tokens.", type=int,
        default=Defaults.TRANSFORMER_HIDDEN_SIZE)
    parser.add_argument(
        "--transformer_intermediate_size", "-tis", help=(
            "Size of the intermediate units in the FFN of the Transformer "
            "encoder."), type=int,
        default=Defaults.TRANSFORMER_INTERMEDIATE_SIZE)
    parser.add_argument(
        "--transformer_attention_heads", "-tah", help=(
            "Number of attention heads in the Transformer encoder."), type=int,
        default=Defaults.TRANSFORMER_ATTENTION_HEADS)
    parser.add_argument(
        "--backbone", "-bb", help="Backbone for models", type=str,
        default=Defaults.BACKBONE,
        choices=[BACKBONE_UNET, BACKBONE_TRANSFORMER_ENCODER])
    parser.add_argument(
        "--skip_init", "-si", help=(
            "Whether to use SkipInit with the residual blocks."),
        type=int, default=Defaults.SKIP_INIT, choices=[0, 1])
    parser.add_argument(
        "--chunk_duration", "-chd", help=(
            "Duration of the video chunks (in seconds)."), type=float,
        default=Defaults.CHUNK_DURATION)
    parser.add_argument(
        "--min_valid_chunk_duration", "-mchd", help=(
            "Minimum duration of the part of a chunk that needs to be valid "
            "(in seconds)."), type=float,
        default=Defaults.MIN_VALID_CHUNK_DURATION)
    parser.add_argument(
        "--nms_window", "-nw", help=(
            "Non-maximum suppression window size to use (in seconds)"),
        type=float, default=Defaults.NMS_WINDOW)
    parser.add_argument(
        "--chunk_prediction_border", "-cbp", help=(
            "During inference, how many seconds should be discarded from each "
            "border."), type=float, default=Defaults.CHUNK_PREDICTION_BORDER)
    parser.add_argument(
        "--validation_chunk_prediction_border", "-vcbp", help=(
            "During validation inference, how many seconds should be discarded "
            "from each border."), type=float,
        default=Defaults.VALIDATION_CHUNK_PREDICTION_BORDER)
    parser.add_argument(
        "--width", "-wi", help=(
            "Width of the models (multiplicative factor for the number "
            "of channels of all layers)"), type=int, default=Defaults.WIDTH)
    parser.add_argument(
        "--reduction_width_factor", "-rwf", help=(
            "Factor that multiplies the width to obtain the number of channels "
            "of the output of the input reduction"), type=int,
        default=Defaults.REDUCTION_WIDTH_FACTOR)
    parser.add_argument(
        "--head_layers", "-hl", help=(
            "Number of convolutional layers in some of the prediction heads."),
        type=int, default=Defaults.HEAD_LAYERS)
    parser.add_argument(
        "--decoupled_weight_decay", "-dwd", help=(
            "Float defining the L2 weight decay applied globally via the "
            "decoupled weight decay extension (0.0 for none)."),
        type=float, default=Defaults.DECOUPLED_WEIGHT_DECAY)
    parser.add_argument(
        "--layer_weight_decay", "-lawd", help=(
            "Float defining the L2 weight decay used when defining individual "
            "layers  (0.0 for no weight decay)."),
        type=float, default=Defaults.LAYER_WEIGHT_DECAY)
    parser.add_argument(
        "--input_weight_decay", "-iwd", help=(
            "Float defining the L2 weight decay (0.0 for no weight decay) to "
            "be used for the initial layers of the models only."),
        type=float, default=Defaults.INPUT_WEIGHT_DECAY)
    parser.add_argument(
        "--batch_norm", "-bn", help=(
            "Whether to use batch-norm layers with convolutions."),
        type=int, default=Defaults.BATCH_NORM, choices=[0, 1])
    parser.add_argument(
        "--dropout", "-do", help=(
            "Dropout rate (rate of elements being dropped out, so higher "
            "means more dropout)."), type=float, default=Defaults.DROPOUT)
    parser.add_argument(
        "--evaluate", help="Whether to run evaluation or just prediction",
        type=int, default=Defaults.EVALUATE, choices=[0, 1])
    parser.add_argument(
        "--create_confusion", "-cc", help=(
            "Whether to create and save more detailed confusion matrix data "
            "during evaluation"), type=int, default=Defaults.CREATE_CONFUSION,
        choices=[0, 1])
    parser.add_argument(
        "--apply_nms", help=(
            "Whether or not to apply non-maximum suppression."), type=int,
        default=Defaults.APPLY_NMS)
    parser.add_argument(
        "--nms_decay", "-nmsd", help=(
            "Which type of decay to apply to the scores within NMS"),
        type=str, default=Defaults.NMS_DECAY,
        choices=[NMS_DECAY_SUPPRESS, NMS_DECAY_LINEAR, NMS_DECAY_GAUSSIAN])
    parser.add_argument(
        "--nms_decay_linear_min", help=(
            "Minimum weight for the soft-NMS linear decay."),
        type=float, default=Defaults.NMS_DECAY_LINEAR_MIN)
    parser.add_argument(
        "--test_split", "-ts", help="Identifier for the split", type=str,
        default=Defaults.TEST_SPLIT, choices=ALL_SPLIT_KEYS)
    parser.add_argument(
        "--prune_classes", "-pc", help=(
            "Flag indicating whether to prune classes with few labels when "
            "running evaluation or validation."), type=int,
        default=Defaults.PRUNE_CLASSES, choices=[0, 1])
    parser.add_argument(
        "--segmentation_evaluation_old", "-seo", help=(
            "Whether to run the old version of segmentation evaluation, "
            "in addition to the new one."), type=int,
        default=Defaults.SEGMENTATION_EVALUATION_OLD, choices=[0, 1])
    parser.add_argument(
        "--test_predict", "-tp", help=(
            "Whether to run inference during testing, as opposed to using "
            "pre-existing saved results"), type=int,
        default=Defaults.TEST_PREDICT, choices=[0, 1])
    parser.add_argument(
        "--test_nms_only", "-tno", help=(
            "Whether to run NMS during testing, when not running inference."),
        type=int, default=Defaults.TEST_NMS_ONLY, choices=[0, 1])
    parser.add_argument(
        "--test_save_labels", "-tsl", help=(
            "Whether to save the labels during testing, when not running "
            "inference."), type=int, default=Defaults.TEST_SAVE_LABELS,
        choices=[0, 1])
    parser.add_argument(
        "--test_save_spotting_jsons", "-tssj", help=(
            "Whether to convert the results to JSON format during testing."),
        type=int, default=Defaults.TEST_SAVE_SPOTTING_JSONS, choices=[0, 1])
    parser.add_argument(
        "--profile", "-pro", help="Whether to profile models inference.",
        type=int, default=Defaults.PROFILE, choices=[0, 1])
    parser.add_argument(
        "--initialize_biases", "-ib",
        help="Whether to initialize the output biases or just use zeros",
        type=int, default=Defaults.INITIALIZE_BIASES, choices=[0, 1])
    parser.add_argument(
        "--focusing_gamma", "-fg",
        help=("Focusing parameter gamma for the focal loss. Set this to 0 in "
              "order to not use the focal loss"), type=float,
        default=Defaults.FOCUSING_GAMMA)
    parser.add_argument(
        "--dense_detection_radius", "-ddr",
        help="Radius used in dense detection reparameterization and loss",
        type=float, default=Defaults.DENSE_DETECTION_RADIUS)
    parser.add_argument(
        "--throw_out_delta", "-tod", help=(
            "Whether to use the predicted deltas during inference."),
        type=int, default=Defaults.THROW_OUT_DELTA, choices=[0, 1])
    parser.add_argument(
        "--delta_radius_multiplier", "-drm", help=(
            "Multiply the dense detection radius by this factor for "
            "determining delta's range"), type=float,
        default=Defaults.DELTA_RADIUS_MULTIPLIER)
    parser.add_argument(
        "--cyclic_delta", "-ccl", help="Whether the delta is cyclic", type=int,
        default=Defaults.CYCLIC_DELTA, choices=[0, 1])
    parser.add_argument(
        "--huber_delta", "-hd", help=(
            "Delta used in the Huber loss. Differences used as input to the "
            "Huber loss are expected to be between 0.0 and 1.0."), type=float,
        default=Defaults.HUBER_DELTA)
    return parser
