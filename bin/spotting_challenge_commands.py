# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import os
from typing import List, Optional

# TODO: figure out how to move code around so that we can avoid importing
#  scripts from bin over here.
from bin.command_user_constants import BAIDU_FEATURES_DIR, \
    BAIDU_TWO_FEATURES_DIR, RESNET_FEATURES_DIR, \
    RESNET_NORMALIZED_FEATURES_DIR, SPLITS_DIR, MODELS_DIR, LABELS_DIR, \
    BASE_CONFIG_DIR, RESULTS_DIR, RUN_NAME, FEATURES_DIR, MEMORY_SETUP, \
    MEMORY_SETUP_256GB, MEMORY_SETUP_64GB
from bin.create_features_from_results import Args as FromResultsArgs
from bin.create_normalizer import Args as CreateNormalizerArgs, \
    NORMALIZER_MAX_ABS
from bin.transform_features import Args as TransformArgs
from spivak.application.argument_parser import \
    DATASET_TYPE_SOCCERNET_V2, DATASET_TYPE_SOCCERNET_V2_CHALLENGE, \
    DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION, DETECTOR_DENSE, \
    DETECTOR_DENSE_DELTA, DETECTOR_AVERAGING_CONFIDENCE
from spivak.application.command_utils import Command, SPOTTING_CHALLENGE, \
    SCRIPT_TRANSFORM, create_name, SCRIPT_TRAIN, SCRIPT_TEST, \
    CONFIG_DIR_CHALLENGE_CONFIDENCE, detector_args, \
    CONFIG_DIR_CHALLENGE_DELTA, CONFIG_DIR_CHALLENGE_DELTA_SOFT_NMS, \
    print_command_list, SCRIPT_CREATE_NORMALIZER, \
    SCRIPT_CREATE_FEATURES_FROM_RESULTS, EXECUTABLE_ZIP, \
    SCRIPT_CREATE_AVERAGING_PREDICTOR, SPOTTING_CHALLENGE_VALIDATED, \
    SPOTTING_TEST, CONFIG_DIR_DELTA_SOFT_NMS, CONFIG_DIR_CONFIDENCE, \
    CONFIG_DIR_DELTA
from spivak.application.validation import LAST_MODEL_DIR, BEST_MODEL_DIR
from spivak.data.dataset_splits import SPLIT_KEY_TEST, SPLIT_KEY_VALIDATION, \
    SPLIT_KEY_UNLABELED
from spivak.data.soccernet_label_io import RESULTS_JSON
from spivak.feature_extraction.extraction import \
    SOCCERNET_FEATURE_NAME_RESNET_TF2
from spivak.models.dense_predictor import OUTPUT_CONFIDENCE as CONFIDENCE, \
    OUTPUT_DELTA as DELTA

NMS_TYPE_20 = "nms_20"
NMS_TYPE_TUNED = "nms_tuned"
NMS_TYPE_SOFT_TUNED = "soft_nms_tuned"
RESNET_NORMALIZER_PKL = "resnet_normalizer.pkl"
RESNET_NORMALIZED_FEATURE_NAME = "resnet_normalized"
BAIDU_TWO_FEATURE_NAME = "baidu_2.0"
# AVERAGED_CONFIDENCE is used only for creating directory names.
AVERAGED_CONFIDENCE = "averaged_confidence"
CONCATENATED_CONFIDENCE_FEATURES_DIR = "concatenated_confidence"
CONCATENATION_FEATURE_NAMES_BAIDU_RESNET = [
    BAIDU_TWO_FEATURE_NAME, RESNET_NORMALIZED_FEATURE_NAME]
MEMORY_TRAIN_PARAMETERS = {
    MEMORY_SETUP_256GB: {
        BAIDU_TWO_FEATURE_NAME: {},
        RESNET_NORMALIZED_FEATURE_NAME: {}
    },
    MEMORY_SETUP_64GB: {
        # TODO: Experiment with higher cpm (chunks per minute) to increase
        #  speed at the cost of less mixing of chunks across videos.
        BAIDU_TWO_FEATURE_NAME: {
            "-cds": "0",    # Don't cache the dataset
            "-cpm": "2.0",  # Sample more chunks each time the features are read
            "-cs": "0.08",  # Use a smaller buffer to shuffle the chunks
            "-sv": "1",     # Shuffle the videos
            "-gcp": "0"     # Remove parallelism from chunk creation
        },
        # We're already under 64GB when training with the normalized ResNet
        # features, so no tweaking is needed here.
        RESNET_NORMALIZED_FEATURE_NAME: {}
    }
}
ENVIRONMENT_VARIABLES_REDUCE_MEMORY = {
    # Setting the variable below addresses an issue with too much memory
    # growth in tensorflow 2.3.0. Unfortunately, it also slows things
    # down. See:
    # https://github.com/tensorflow/tensorflow/issues/44176
    # In practice, using 0 seemed faster than 128 * 1024, even though both
    # worked to stop the memory growth.
    "MALLOC_TRIM_THRESHOLD_": "0"
}
MEMORY_TRAIN_ENVIRONMENT_VARIABLES = {
    MEMORY_SETUP_256GB: {
        BAIDU_TWO_FEATURE_NAME: {},
        RESNET_NORMALIZED_FEATURE_NAME: {}
    },
    MEMORY_SETUP_64GB: {
        BAIDU_TWO_FEATURE_NAME: ENVIRONMENT_VARIABLES_REDUCE_MEMORY,
        RESNET_NORMALIZED_FEATURE_NAME: ENVIRONMENT_VARIABLES_REDUCE_MEMORY
    }
}
DELTA_TEST_PARAMETERS_PER_SPLIT = {
    SPLIT_KEY_TEST: {},
    SPLIT_KEY_VALIDATION: {},
    SPLIT_KEY_UNLABELED: {
        "-tssj": "1"
    }
}
DELTA_TEST_PARAMETERS_PER_NMS_TYPE = {
    NMS_TYPE_20: {},
    NMS_TYPE_TUNED: {},
    NMS_TYPE_SOFT_TUNED: {
        "-nmsd": "linear"
    }
}
TRAIN_HYPERPARAMETERS_CHALLENGE_VALIDATED_BAIDU_2 = {
    CONFIDENCE: {
        "-lr": "5e-5",
        "-dwd": "5e-5",
        "-sr": "0.0",
        "-mu": "0.0",
    },
    DELTA: {
        "-lr": "2e-3",
        "-dwd": "1e-3",
        "-sr": "0.1"
    }
}
TRAIN_HYPERPARAMETERS = {
    DATASET_TYPE_SOCCERNET_V2: {
        BAIDU_TWO_FEATURE_NAME: {
            CONFIDENCE: {
                "-lr": "2e-4",
                "-dwd": "2e-4",
                "-sr": "0.0",
                "-mu": "2.0"
            },
            DELTA: {
                "-lr": "5e-4",
                "-dwd": "5e-4",
                "-sr": "0.7"
            },
        },
        RESNET_NORMALIZED_FEATURE_NAME: {
            CONFIDENCE: {
                "-lr": "1e-3",
                "-dwd": "2e-4",
                "-sr": "0.5",
                "-mu": "2.0"
            },
            DELTA: {
                "-lr": "1e-3",
                "-dwd": "5e-4",
                "-sr": "0.5"
            }
        }
    },
    DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION: {
        BAIDU_TWO_FEATURE_NAME:
            TRAIN_HYPERPARAMETERS_CHALLENGE_VALIDATED_BAIDU_2,
        RESNET_NORMALIZED_FEATURE_NAME: {
            CONFIDENCE: {
                "-lr": "5e-4",
                "-dwd": "2e-4",
                "-sr": "0.02",
                "-mu": "0.0"
            },
            DELTA: {
                "-lr": "5e-4",
                "-dwd": "2e-4",
                "-sr": "0.7"
            },
        },
    },
    DATASET_TYPE_SOCCERNET_V2_CHALLENGE: {
        # For the baidu_2.0 features, in the Challenge protocol, we directly
        # reused the hyperparameters that were found using the Challenge
        # Validated protocol. In this particular case, we didn't try to fiddle
        # with the number of training epochs since we didn't notice much
        # overfitting when doing the equivalent experiments with the validation
        # set.
        BAIDU_TWO_FEATURE_NAME:
            TRAIN_HYPERPARAMETERS_CHALLENGE_VALIDATED_BAIDU_2,
        RESNET_NORMALIZED_FEATURE_NAME: {
            # Our training runs for the confidence score with the ResNet
            # normalized features would show some overfitting when reaching a
            # higher number of epochs of the Challenge Validated protocol. In
            # the Challenge protocol, we train with all the available labelled
            # data, so we do not have a validation set with which to pick the
            # best model. So we ended up guessing a good number of epochs to
            # use in this case, based on an extrapolation of the results from
            # the Challenge Validated protocol.
            CONFIDENCE: {
                "-lr": "5e-4",
                "-dwd": "2e-4",
                "-sr": "0.02",
                "-mu": "0.0",
                "-e": "500"
            },
            DELTA: {
                "-lr": "5e-4",
                "-dwd": "2e-4",
                "-sr": "0.7",
                "-e": "1250",
                "-lrdle": "1250"
            }
        }
    }
}


def command_resample_baidu(
        baidu_features_dir: str = BAIDU_FEATURES_DIR,
        baidu_two_features_dir: str = BAIDU_TWO_FEATURES_DIR
) -> Command:
    return Command(
        "Resample baidu features to 2.0 frames per second", SCRIPT_TRANSFORM,
        {
            f"--{TransformArgs.INPUT_DIRS}": baidu_features_dir,
            f"--{TransformArgs.INPUT_FEATURE_NAMES}": "baidu_soccer_embeddings",
            f"--{TransformArgs.OUTPUT_DIR}": baidu_two_features_dir,
            f"--{TransformArgs.OUTPUT_FEATURE_NAME}": BAIDU_TWO_FEATURE_NAME,
            f"--{TransformArgs.FACTORS}": "2.0",
            f"--{TransformArgs.RESAMPLING}": "interpolate"
        }
    )


def commands_normalize_resnet(
        resnet_features_dir: str = RESNET_FEATURES_DIR,
        resnet_normalized_features_dir: str = RESNET_NORMALIZED_FEATURES_DIR,
        models_dir: str = MODELS_DIR,
        splits_dir: str = SPLITS_DIR
) -> List[Command]:
    normalizer_path = os.path.join(models_dir, RESNET_NORMALIZER_PKL)
    create_normalizer_command = Command(
        "Create a normalizer for the ResNet features", SCRIPT_CREATE_NORMALIZER,
        {
            f"--{CreateNormalizerArgs.FEATURES_DIR}": resnet_features_dir,
            f"--{CreateNormalizerArgs.SPLITS_DIR}": splits_dir,
            f"--{CreateNormalizerArgs.NORMALIZER}": NORMALIZER_MAX_ABS,
            f"--{CreateNormalizerArgs.FEATURE_NAME}":
                SOCCERNET_FEATURE_NAME_RESNET_TF2,
            f"--{CreateNormalizerArgs.OUT_PATH}": normalizer_path,
        }
    )
    normalize_features_command = Command(
        "Normalize the ResNet features with the learned normalizer",
        SCRIPT_TRANSFORM,
        {
            f"--{TransformArgs.INPUT_DIRS}": resnet_features_dir,
            f"--{TransformArgs.INPUT_FEATURE_NAMES}":
                SOCCERNET_FEATURE_NAME_RESNET_TF2,
            f"--{TransformArgs.OUTPUT_DIR}": resnet_normalized_features_dir,
            f"--{TransformArgs.OUTPUT_FEATURE_NAME}":
                RESNET_NORMALIZED_FEATURE_NAME,
            f"--{TransformArgs.NORMALIZERS}": normalizer_path,
            f"--{TransformArgs.FACTORS}": "1.0",  # Won't resample
            f"--{TransformArgs.RESAMPLING}": "interpolate",
        }
    )
    return [create_normalizer_command, normalize_features_command]


def commands_spotting_test(
        specific_features_dir: str,
        feature_name: str,
        do_nms_comparison: bool = False,
        run_name: str = RUN_NAME,
        results_dir: str = RESULTS_DIR,
        models_dir: str = MODELS_DIR,
        labels_dir: str = LABELS_DIR,
        splits_dir: str = SPLITS_DIR,
        base_config_dir: str = BASE_CONFIG_DIR,
        memory_setup: str = MEMORY_SETUP
) -> List[Command]:
    dataset_type = DATASET_TYPE_SOCCERNET_V2
    protocol_name = SPOTTING_TEST
    confidence_and_delta_validated_train_commands = \
        _commands_confidence_and_delta_validated_train(
            memory_setup, dataset_type, protocol_name, specific_features_dir,
            feature_name, run_name, results_dir, models_dir, labels_dir,
            splits_dir, base_config_dir)
    if do_nms_comparison:
        nms_types = [NMS_TYPE_20, NMS_TYPE_TUNED, NMS_TYPE_SOFT_TUNED]
    else:
        nms_types = [NMS_TYPE_SOFT_TUNED]
    last_test_commands = []
    for nms_type in nms_types:
        # Run testing on both the validation (SPLIT_KEY_VALIDATION) and test
        # (SPLIT_KEY_TEST) splits.
        for split_key in [SPLIT_KEY_VALIDATION, SPLIT_KEY_TEST]:
            confidence_and_delta_test_commands = \
                _commands_spotting_confidence_and_delta_test(
                    split_key, nms_type, dataset_type, protocol_name,
                    specific_features_dir, feature_name, run_name,
                    results_dir, models_dir, labels_dir, splits_dir,
                    base_config_dir)
            last_test_commands.extend(confidence_and_delta_test_commands)
    return confidence_and_delta_validated_train_commands + last_test_commands


def commands_spotting_challenge_validated(
        specific_features_dir: str,
        feature_name: str,
        run_name: str = RUN_NAME,
        results_dir: str = RESULTS_DIR,
        models_dir: str = MODELS_DIR,
        labels_dir: str = LABELS_DIR,
        splits_dir: str = SPLITS_DIR,
        base_config_dir: str = BASE_CONFIG_DIR,
        memory_setup: str = MEMORY_SETUP
) -> List[Command]:
    dataset_type = DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION
    protocol_name = SPOTTING_CHALLENGE_VALIDATED
    confidence_and_delta_validated_train_commands = \
        _commands_confidence_and_delta_validated_train(
            memory_setup, dataset_type, protocol_name, specific_features_dir,
            feature_name, run_name, results_dir, models_dir, labels_dir,
            splits_dir, base_config_dir)
    # Run testing on both the validation (SPLIT_KEY_VALIDATION) and challenge
    # (SPLIT_KEY_UNLABELED) splits.
    last_test_commands = []
    for split_key in [SPLIT_KEY_VALIDATION, SPLIT_KEY_UNLABELED]:
        confidence_and_delta_test_commands = \
            _commands_spotting_confidence_and_delta_test(
                split_key, NMS_TYPE_SOFT_TUNED, dataset_type, protocol_name,
                specific_features_dir, feature_name, run_name, results_dir,
                models_dir, labels_dir, splits_dir, base_config_dir)
        last_test_commands.extend(confidence_and_delta_test_commands)
    return confidence_and_delta_validated_train_commands + last_test_commands


def commands_spotting_challenge(
        specific_features_dir: str,
        feature_name: str,
        run_name: str = RUN_NAME,
        results_dir: str = RESULTS_DIR,
        models_dir: str = MODELS_DIR,
        labels_dir: str = LABELS_DIR,
        splits_dir: str = SPLITS_DIR,
        base_config_dir: str = BASE_CONFIG_DIR,
        memory_setup: str = MEMORY_SETUP
) -> List[Command]:
    dataset_type = DATASET_TYPE_SOCCERNET_V2_CHALLENGE
    protocol_name = SPOTTING_CHALLENGE
    confidence_train_command = _command_spotting_confidence_train(
        specific_features_dir, feature_name, dataset_type, protocol_name,
        run_name, models_dir, labels_dir, splits_dir, base_config_dir,
        memory_setup)
    delta_train_command = _command_spotting_delta_train(
        None, specific_features_dir, feature_name, dataset_type, protocol_name,
        run_name, models_dir, labels_dir, splits_dir, base_config_dir,
        memory_setup)
    confidence_and_delta_test_commands = \
        _commands_spotting_confidence_and_delta_test(
            SPLIT_KEY_UNLABELED, NMS_TYPE_SOFT_TUNED, dataset_type,
            protocol_name, specific_features_dir, feature_name, run_name,
            results_dir, models_dir, labels_dir, splits_dir, base_config_dir)
    return [
        confidence_train_command, delta_train_command,
        *confidence_and_delta_test_commands]


def commands_spotting_test_fusion(
        baidu_two_features_dir: str = BAIDU_TWO_FEATURES_DIR,
        run_name: str = RUN_NAME,
        results_dir: str = RESULTS_DIR,
        models_dir: str = MODELS_DIR,
        features_dir: str = FEATURES_DIR,
        labels_dir: str = LABELS_DIR,
        splits_dir: str = SPLITS_DIR,
        base_config_dir: str = BASE_CONFIG_DIR
) -> List[Command]:
    return _commands_spotting_fusion_train_and_test(
        [SPLIT_KEY_VALIDATION, SPLIT_KEY_TEST], DATASET_TYPE_SOCCERNET_V2,
        SPOTTING_TEST, baidu_two_features_dir, run_name, results_dir,
        models_dir, features_dir, labels_dir, splits_dir, base_config_dir)


def commands_spotting_challenge_validated_fusion(
        baidu_two_features_dir: str = BAIDU_TWO_FEATURES_DIR,
        run_name: str = RUN_NAME,
        results_dir: str = RESULTS_DIR,
        models_dir: str = MODELS_DIR,
        features_dir: str = FEATURES_DIR,
        labels_dir: str = LABELS_DIR,
        splits_dir: str = SPLITS_DIR,
        base_config_dir: str = BASE_CONFIG_DIR
) -> List[Command]:
    return _commands_spotting_fusion_train_and_test(
        [SPLIT_KEY_VALIDATION, SPLIT_KEY_UNLABELED],
        DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION,
        SPOTTING_CHALLENGE_VALIDATED, baidu_two_features_dir, run_name,
        results_dir, models_dir, features_dir, labels_dir, splits_dir,
        base_config_dir)


def commands_spotting_challenge_fusion(
        baidu_two_features_dir: str = BAIDU_TWO_FEATURES_DIR,
        run_name: str = RUN_NAME,
        results_dir: str = RESULTS_DIR,
        models_dir: str = MODELS_DIR,
        features_dir: str = FEATURES_DIR,
        labels_dir: str = LABELS_DIR,
        splits_dir: str = SPLITS_DIR,
        base_config_dir: str = BASE_CONFIG_DIR
) -> List[Command]:
    protocol_name = SPOTTING_CHALLENGE
    dataset_type = DATASET_TYPE_SOCCERNET_V2_CHALLENGE
    split_key = SPLIT_KEY_UNLABELED
    features_from_confidences_command = \
        _command_features_from_confidence_results(
            split_key, CONCATENATION_FEATURE_NAMES_BAIDU_RESNET, dataset_type,
            protocol_name, run_name, results_dir, features_dir)
    concatenated_confidence_features_dir = \
        _concatenated_confidence_features_dir(
            protocol_name, CONCATENATION_FEATURE_NAMES_BAIDU_RESNET,
            features_dir)
    # Confidence averaging uses the existing model from the Challenge Validated
    # protocol, since we don't have a validation set in the Challenge protocol.
    confidence_averaging_model_file = \
        _spotting_challenge_validated_confidence_averaging_model_file(
            SPOTTING_CHALLENGE_VALIDATED,
            CONCATENATION_FEATURE_NAMES_BAIDU_RESNET, run_name, models_dir)
    test_commands = _commands_spotting_confidence_averaging_and_delta_test(
        split_key, concatenated_confidence_features_dir,
        confidence_averaging_model_file, dataset_type, protocol_name,
        run_name, results_dir, models_dir, labels_dir, splits_dir,
        base_config_dir, baidu_two_features_dir)
    return [features_from_confidences_command, *test_commands]


# We define print_commands here so that users can import it from the current
# module (spotting_challenge_commands.py).
print_commands = print_command_list


def _commands_confidence_and_delta_validated_train(
        memory_setup: str, dataset_type: str, protocol_name: str,
        specific_features_dir: str, feature_name: str, run_name: str,
        results_dir: str, models_dir: str, labels_dir: str, splits_dir: str,
        base_config_dir: str) -> List[Command]:
    confidence_train_command = _command_spotting_confidence_train(
        specific_features_dir, feature_name, dataset_type, protocol_name,
        run_name, models_dir, labels_dir, splits_dir, base_config_dir,
        memory_setup)
    # First, we run the confidence model on the validation split, so that the
    # confidence scores can be used during the validation step when training
    # the delta model below.
    confidence_validation_results_dir = _spotting_confidence_results_dir(
        feature_name, SPLIT_KEY_VALIDATION, dataset_type, protocol_name,
        run_name, results_dir)
    confidence_test_validation_command = _command_spotting_confidence_test(
        confidence_validation_results_dir, specific_features_dir, feature_name,
        SPLIT_KEY_VALIDATION, dataset_type, protocol_name, run_name, models_dir,
        labels_dir, splits_dir, base_config_dir)
    delta_train_command = _command_spotting_delta_train(
        confidence_validation_results_dir, specific_features_dir, feature_name,
        dataset_type, protocol_name, run_name, models_dir, labels_dir,
        splits_dir, base_config_dir, memory_setup)
    return [
        confidence_train_command, confidence_test_validation_command,
        delta_train_command]


def _commands_spotting_confidence_and_delta_test(
        split_key: str, nms_type: str, dataset_type: str, protocol_name: str,
        specific_features_dir: str, feature_name: str, run_name: str,
        results_dir: str, models_dir: str, labels_dir: str, splits_dir: str,
        base_config_dir: str) -> List[Command]:
    confidence_and_delta_results_dir = \
        _spotting_confidence_and_delta_results_dir(
            CONFIDENCE, feature_name, split_key, nms_type, dataset_type,
            protocol_name, run_name, results_dir)
    confidence_test_command = _command_spotting_confidence_test(
        confidence_and_delta_results_dir, specific_features_dir,
        feature_name, split_key, dataset_type, protocol_name, run_name,
        models_dir, labels_dir, splits_dir, base_config_dir)
    delta_test_commands = _commands_spotting_delta_test(
        confidence_and_delta_results_dir, specific_features_dir,
        feature_name, split_key, nms_type, dataset_type, protocol_name,
        run_name, models_dir, labels_dir, splits_dir, base_config_dir)
    return [confidence_test_command] + delta_test_commands


def _commands_spotting_fusion_train_and_test(
        split_keys_test: List[str], dataset_type: str, protocol_name: str,
        baidu_two_features_dir: str, run_name: str, results_dir: str,
        models_dir: str, features_dir: str, labels_dir: str, splits_dir: str,
        base_config_dir: str) -> List[Command]:
    # Concatenate the existing Baidu and ResNet confidence scores for the
    # splits in split_keys_test.
    features_from_confidences_commands = [
        _command_features_from_confidence_results(
            split_key, CONCATENATION_FEATURE_NAMES_BAIDU_RESNET, dataset_type,
            protocol_name, run_name, results_dir, features_dir)
        for split_key in split_keys_test
    ]
    concatenated_confidence_features_dir = \
        _concatenated_confidence_features_dir(
            protocol_name, CONCATENATION_FEATURE_NAMES_BAIDU_RESNET,
            features_dir)
    confidence_averaging_model_file = \
        _spotting_challenge_validated_confidence_averaging_model_file(
            protocol_name, CONCATENATION_FEATURE_NAMES_BAIDU_RESNET, run_name,
            models_dir)
    confidence_averaging_train_command = \
        _command_spotting_confidence_averaging_train(
            concatenated_confidence_features_dir,
            confidence_averaging_model_file, dataset_type, labels_dir,
            splits_dir, base_config_dir)
    test_commands = []
    for split_key in split_keys_test:
        test_commands.extend(
            _commands_spotting_confidence_averaging_and_delta_test(
                split_key, concatenated_confidence_features_dir,
                confidence_averaging_model_file, dataset_type, protocol_name,
                run_name, results_dir, models_dir, labels_dir, splits_dir,
                base_config_dir, baidu_two_features_dir
            )
        )
    return [
        *features_from_confidences_commands,
        confidence_averaging_train_command, *test_commands]


def _commands_spotting_confidence_averaging_and_delta_test(
        split_key: str, concatenated_confidence_features_dir: str,
        confidence_averaging_model_file: str, dataset_type: str,
        protocol_name: str, run_name: str, results_dir: str, models_dir: str,
        labels_dir: str, splits_dir: str, base_config_dir: str,
        baidu_two_features_dir: str) -> List[Command]:
    # The confidence scores we're using are obained from fusing the
    # confidence scores resulting from different models. However,
    # when inferring the deltas, we use only the model trained on the Baidu
    # features when inferring, since we didn't see any benefit in fusing the
    # delta scores.
    confidence_and_delta_results_dir = \
        _spotting_confidence_and_delta_results_dir(
            AVERAGED_CONFIDENCE, BAIDU_TWO_FEATURE_NAME, split_key,
            NMS_TYPE_SOFT_TUNED, dataset_type, protocol_name, run_name,
            results_dir)
    confidence_averaging_test_command = \
        _command_spotting_confidence_averaging_test(
            confidence_and_delta_results_dir,
            concatenated_confidence_features_dir,
            confidence_averaging_model_file, split_key, dataset_type,
            protocol_name, labels_dir, splits_dir, base_config_dir)
    delta_test_commands = _commands_spotting_delta_test(
        confidence_and_delta_results_dir, baidu_two_features_dir,
        BAIDU_TWO_FEATURE_NAME, split_key, NMS_TYPE_SOFT_TUNED, dataset_type,
        protocol_name, run_name, models_dir, labels_dir, splits_dir,
        base_config_dir)
    return [confidence_averaging_test_command, *delta_test_commands]


def _command_spotting_confidence_train(
        specific_features_dir: str, feature_name: str, dataset_type: str,
        protocol_name: str, run_name: str, models_dir: str, labels_dir: str,
        splits_dir: str, base_config_dir: str, memory_setup: str) -> Command:
    confidence_train_hyperparameters = TRAIN_HYPERPARAMETERS[dataset_type][
        feature_name][CONFIDENCE]
    confidence_model_dir = os.path.join(
        models_dir,
        create_name(
            confidence_train_hyperparameters, run_name, CONFIDENCE,
            feature_name, protocol_name)
    )
    config_dir = _confidence_config_dir(base_config_dir, dataset_type)
    return Command(
        f"Train the confidence model on the {dataset_type} dataset ("
        f"{protocol_name} protocol) on the {feature_name} features",
        SCRIPT_TRAIN,
        {
            "-sd": splits_dir,
            "-ld": labels_dir,
            "-fd": specific_features_dir,
            "-fn": feature_name,
            "-dt": dataset_type,
            "-cd": config_dir,
            **MEMORY_TRAIN_PARAMETERS[memory_setup][feature_name],
            **detector_args(DETECTOR_DENSE),
            **confidence_train_hyperparameters,
            "-m": confidence_model_dir
        },
        env_vars=MEMORY_TRAIN_ENVIRONMENT_VARIABLES[memory_setup][feature_name]
    )


def _command_spotting_confidence_test(
        specific_results_dir: str, specific_features_dir: str,
        feature_name: str, split_key: str, dataset_type: str,
        protocol_name: str, run_name: str, models_dir: str, labels_dir: str,
        splits_dir: str, base_config_dir: str) -> Command:
    confidence_train_hyperparameters = TRAIN_HYPERPARAMETERS[dataset_type][
        feature_name][CONFIDENCE]
    confidence_model_dir = os.path.join(
        models_dir,
        create_name(
            confidence_train_hyperparameters, run_name, CONFIDENCE,
            feature_name, protocol_name)
    )
    config_dir = _confidence_config_dir(base_config_dir, dataset_type)
    model_last_or_best = _model_last_or_best(dataset_type)
    return Command(
        f"Test the confidence model on the {split_key} split of "
        f"the {dataset_type} dataset ({protocol_name} protocol) on the"
        f" {feature_name} features",
        SCRIPT_TEST,
        {
            "-sd": splits_dir,
            "-ld": labels_dir,
            "-fd": specific_features_dir,
            "-fn": feature_name,
            "-dt": dataset_type,
            "-cd": config_dir,
            **detector_args(DETECTOR_DENSE),
            "-ts": split_key,
            "-m": os.path.join(confidence_model_dir, model_last_or_best),
            "-rd": specific_results_dir
        }
    )


def _command_spotting_delta_train(
        confidence_validation_results_dir: Optional[str],
        specific_features_dir: str, feature_name: str, dataset_type: str,
        protocol_name: str, run_name: str, models_dir: str, labels_dir: str,
        splits_dir: str, base_config_dir: str, memory_setup: str) -> Command:
    delta_train_hyperparameters = TRAIN_HYPERPARAMETERS[dataset_type][
        feature_name][DELTA]
    delta_model_dir = _spotting_delta_model_dir(
        feature_name, dataset_type, protocol_name, run_name, models_dir)
    config_dir = _delta_config_dir(
        base_config_dir, dataset_type, NMS_TYPE_TUNED)
    command_arguments = {
        "-sd": splits_dir,
        "-ld": labels_dir,
        "-fd": specific_features_dir,
        "-fn": feature_name,
        "-dt": dataset_type,
        "-cd": config_dir,
        **MEMORY_TRAIN_PARAMETERS[memory_setup][feature_name],
        **detector_args(DETECTOR_DENSE_DELTA),
        **delta_train_hyperparameters,
        "-m": delta_model_dir,
    }
    if confidence_validation_results_dir:
        command_arguments["-rd"] = confidence_validation_results_dir
    return Command(
        f"Train the displacement (delta) model on the {dataset_type} dataset "
        f"({protocol_name} protocol) using {feature_name} features",
        SCRIPT_TRAIN,
        command_arguments,
        env_vars=MEMORY_TRAIN_ENVIRONMENT_VARIABLES[memory_setup][feature_name]
    )


def _commands_spotting_delta_test(
        specific_results_dir: str, specific_features_dir: str,
        feature_name: str, split_key: str, nms_type: str, dataset_type: str,
        protocol_name: str, run_name: str, models_dir: str, labels_dir: str,
        splits_dir: str, base_config_dir: str) -> List[Command]:
    delta_model_dir = _spotting_delta_model_dir(
        feature_name, dataset_type, protocol_name, run_name, models_dir)
    model_last_or_best = _model_last_or_best(dataset_type)
    config_dir = _delta_config_dir(base_config_dir, dataset_type, nms_type)
    delta_test_command = Command(
        f"Test the displacement (delta) model on the {split_key} split of the "
        f"{dataset_type} dataset ({protocol_name} protocol) using"
        f" {feature_name} features and post-processing with {nms_type}",
        SCRIPT_TEST,
        {
            "-sd": splits_dir,
            "-ld": labels_dir,
            "-fd": specific_features_dir,
            "-fn": feature_name,
            "-dt": dataset_type,
            "-cd": config_dir,
            **detector_args(DETECTOR_DENSE_DELTA),
            "-tod": "0",
            **DELTA_TEST_PARAMETERS_PER_SPLIT[split_key],
            **DELTA_TEST_PARAMETERS_PER_NMS_TYPE[nms_type],
            "-ts": split_key,
            "-m": os.path.join(delta_model_dir, model_last_or_best),
            "-rd": specific_results_dir
        }
    )
    commands = [delta_test_command]
    if split_key == SPLIT_KEY_UNLABELED:
        zip_results_command = _command_zip_results(specific_results_dir)
        commands.append(zip_results_command)
    return commands


def _command_zip_results(experiment_results_dir: str) -> Command:
    return Command(
        "Zip the spotting result JSONs into a single zip file", EXECUTABLE_ZIP,
        ["-r", "results_spotting.zip", ".", "-i", f"*/*/*/{RESULTS_JSON}"],
        cwd=experiment_results_dir
    )


def _command_features_from_confidence_results(
        split_key: str, feature_names: List[str], dataset_type: str,
        protocol_name: str, run_name: str, results_dir: str,
        features_dir: str) -> Command:
    concatenated_confidence_features_dir = \
        _concatenated_confidence_features_dir(
            protocol_name, feature_names, features_dir)
    results_dirs = [
        _spotting_confidence_and_delta_results_dir(
            CONFIDENCE, feature_name, split_key, NMS_TYPE_SOFT_TUNED,
            dataset_type, protocol_name, run_name, results_dir)
        for feature_name in feature_names
    ]
    results_dirs_str = " ".join(results_dirs)
    return Command(
        "For each video, read the predicted confidence probabilities from the "
        f"models trained on features {feature_names} on the {split_key} split "
        f"of the {dataset_type} dataset ({protocol_name} protocol) and "
        f"concatenate them into a single feature file",
        SCRIPT_CREATE_FEATURES_FROM_RESULTS,
        {
            f"--{FromResultsArgs.INPUT_DIRS}": results_dirs_str,
            f"--{FromResultsArgs.OUTPUT_DIR}":
                concatenated_confidence_features_dir,
            f"--{FromResultsArgs.OUTPUT_NAME}": CONFIDENCE
        }
    )


def _command_spotting_confidence_averaging_train(
        concatenated_confidence_features_dir: str,
        confidence_averaging_model_file: str, dataset_type: str,
        labels_dir: str, splits_dir: str, base_config_dir: str) -> Command:
    config_dir = _confidence_config_dir(base_config_dir, dataset_type)
    return Command(
        "Train a model that averages the confidence probabilities given in the "
        "input features files", SCRIPT_CREATE_AVERAGING_PREDICTOR,
        {
            "-sd": splits_dir,
            "-ld": labels_dir,
            "-fd": concatenated_confidence_features_dir,
            "-fn": CONFIDENCE,
            "-dt": dataset_type,
            "-cd": config_dir,
            **detector_args(DETECTOR_AVERAGING_CONFIDENCE),
            "-m": confidence_averaging_model_file
        }
    )


def _command_spotting_confidence_averaging_test(
        specific_results_dir: str, concatenated_confidence_features_dir: str,
        confidence_averaging_model_file: str, split_key: str, dataset_type: str,
        protocol_name: str, labels_dir: str, splits_dir: str,
        base_config_dir: str) -> Command:
    config_dir = _confidence_config_dir(base_config_dir, dataset_type)
    return Command(
        f"Fuse the confidence scores using the averaging predictor over the "
        f"{split_key} split of the {dataset_type} dataset ({protocol_name} "
        f"protocol)", SCRIPT_TEST,
        {
            "-sd": splits_dir,
            "-ld": labels_dir,
            "-fd": concatenated_confidence_features_dir,
            "-fn": CONFIDENCE,
            "-dt": dataset_type,
            "-cd": config_dir,
            **detector_args(DETECTOR_AVERAGING_CONFIDENCE),
            "-ts": split_key,
            "-m": confidence_averaging_model_file,
            "-rd": specific_results_dir,
        }
    )


def _concatenated_confidence_features_dir(
        protocol_name: str, feature_names: List[str], features_dir: str) -> str:
    feature_names_str = "_".join(feature_names)
    return os.path.join(
        features_dir,
        f"{protocol_name}_{CONCATENATED_CONFIDENCE_FEATURES_DIR}_"
        f"{feature_names_str}")


def _spotting_challenge_validated_confidence_averaging_model_file(
        protocol_name: str, feature_names: List[str], run_name: str,
        models_dir: str) -> str:
    feature_names_str = "_".join(feature_names)
    averaging_model_file_name = (
        f"{protocol_name}_{DETECTOR_AVERAGING_CONFIDENCE}_{feature_names_str}"
        f"_{run_name}.pkl")
    return os.path.join(models_dir, averaging_model_file_name)


def _spotting_confidence_results_dir(
        feature_name: str, split_key: str, dataset_type: str,
        protocol_name: str, run_name: str, results_dir: str) -> str:
    confidence_train_hyperparameters = TRAIN_HYPERPARAMETERS[
        dataset_type][feature_name][CONFIDENCE]
    extra_name = f"{split_key}_{run_name}"
    return os.path.join(
        results_dir,
        create_name(
            confidence_train_hyperparameters, extra_name, CONFIDENCE,
            feature_name, protocol_name)
    )


def _spotting_confidence_and_delta_results_dir(
        confidence_name: str, feature_name: str, split_key: str, nms_type: str,
        dataset_type: str, protocol_name: str, run_name: str,
        results_dir: str) -> str:
    delta_train_hyperparameters = TRAIN_HYPERPARAMETERS[dataset_type][
        feature_name][DELTA]
    delta_test_split_parameters = DELTA_TEST_PARAMETERS_PER_SPLIT[split_key]
    delta_all_parameters = {
        **delta_train_hyperparameters,
        **delta_test_split_parameters}
    extra_name = f"{confidence_name}_{split_key}_{nms_type}_{run_name}"
    return os.path.join(
        results_dir,
        create_name(
            delta_all_parameters, extra_name, DELTA, feature_name,
            protocol_name)
    )


def _spotting_delta_model_dir(
        feature_name: str, dataset_type: str, protocol_name: str, run_name: str,
        models_dir: str) -> str:
    delta_train_hyperparameters = TRAIN_HYPERPARAMETERS[dataset_type][
        feature_name][DELTA]
    return os.path.join(
        models_dir,
        create_name(
            delta_train_hyperparameters, run_name, DELTA, feature_name,
            protocol_name)
    )


def _delta_config_dir(
        base_config_dir: str, dataset_type: str, nms_type: str) -> str:
    if dataset_type in {
        DATASET_TYPE_SOCCERNET_V2_CHALLENGE,
        DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION
    }:
        if nms_type == NMS_TYPE_TUNED:
            dir_name = CONFIG_DIR_CHALLENGE_DELTA
        elif nms_type == NMS_TYPE_SOFT_TUNED:
            dir_name = CONFIG_DIR_CHALLENGE_DELTA_SOFT_NMS
        elif nms_type == NMS_TYPE_20:
            # Ideally, we would have a specific config dir for delta with an
            # NMS window of size 20, but instead here we just re-use the
            # confidence config dir (which uses the NMS window of size 20).
            dir_name = CONFIG_DIR_CHALLENGE_CONFIDENCE
        else:
            raise ValueError(f"Unknown nms_type: {nms_type}")
    elif dataset_type == DATASET_TYPE_SOCCERNET_V2:
        if nms_type == NMS_TYPE_TUNED:
            dir_name = CONFIG_DIR_DELTA
        elif nms_type == NMS_TYPE_SOFT_TUNED:
            dir_name = CONFIG_DIR_DELTA_SOFT_NMS
        elif nms_type == NMS_TYPE_20:
            # Ideally, we would have a specific config dir for delta with an
            # NMS window of size 20, but instead here we just re-use the
            # confidence config dir (which uses the NMS window of size 20).
            dir_name = CONFIG_DIR_CONFIDENCE
        else:
            raise ValueError(f"Unknown nms_type: {nms_type}")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    return os.path.join(base_config_dir, dir_name)


def _confidence_config_dir(base_config_dir: str, dataset_type: str) -> str:
    if dataset_type in {
        DATASET_TYPE_SOCCERNET_V2_CHALLENGE,
        DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION
    }:
        dir_name = CONFIG_DIR_CHALLENGE_CONFIDENCE
    elif dataset_type == DATASET_TYPE_SOCCERNET_V2:
        dir_name = CONFIG_DIR_CONFIDENCE
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    return os.path.join(base_config_dir, dir_name)


def _model_last_or_best(dataset_type: str) -> str:
    # The challenge protocol is the only one which does not run validation,
    # so it does not have the best model directory. In this case, we use the
    # last available model instead.
    if dataset_type == DATASET_TYPE_SOCCERNET_V2_CHALLENGE:
        return LAST_MODEL_DIR
    return BEST_MODEL_DIR
