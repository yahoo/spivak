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
    DATASET_TYPE_SOCCERNET_V2_CHALLENGE, DETECTOR_DENSE, DETECTOR_DENSE_DELTA, \
    DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION, \
    DETECTOR_AVERAGING_CONFIDENCE
from spivak.application.command_utils import Command, SPOTTING_CHALLENGE, \
    SCRIPT_TRANSFORM, create_name, SCRIPT_TRAIN, SCRIPT_TEST, \
    CONFIG_DIR_CHALLENGE_CONFIDENCE, detector_args, \
    CONFIG_DIR_CHALLENGE_DELTA, CONFIG_DIR_CHALLENGE_DELTA_SOFT_NMS, \
    print_command_list, SCRIPT_CREATE_NORMALIZER, \
    SCRIPT_CREATE_FEATURES_FROM_RESULTS, EXECUTABLE_ZIP, \
    SCRIPT_CREATE_AVERAGING_PREDICTOR, SPOTTING_CHALLENGE_VALIDATED
from spivak.application.validation import LAST_MODEL_DIR, BEST_MODEL_DIR
from spivak.data.dataset_splits import SPLIT_KEY_VALIDATION, SPLIT_KEY_UNLABELED
from spivak.data.soccernet_label_io import RESULTS_JSON
from spivak.feature_extraction.extraction import \
    SOCCERNET_FEATURE_NAME_RESNET_TF2
from spivak.models.dense_predictor import OUTPUT_CONFIDENCE as CONFIDENCE, \
    OUTPUT_DELTA as DELTA

RESNET_NORMALIZER_PKL = "resnet_normalizer.pkl"
RESNET_NORMALIZED_FEATURE_NAME = "resnet_normalized"
BAIDU_TWO_FEATURE_NAME = "baidu_2.0"
# AVERAGED_CONFIDENCE is used only for creating directory names.
AVERAGED_CONFIDENCE = "averaged_confidence"
CONCATENATED_CONFIDENCE_FEATURES_DIR = "concatenated_confidence"
MEMORY_TRAIN_PARAMETERS = {
    MEMORY_SETUP_256GB: {
        BAIDU_TWO_FEATURE_NAME: {},
        RESNET_NORMALIZED_FEATURE_NAME: {}
    },
    MEMORY_SETUP_64GB: {
        BAIDU_TWO_FEATURE_NAME: {
            "-cds": "0",    # Don't cache the dataset
            "-cpm": "2.0",  # Sample more chunks each time the features are read
            "-cs": "0.08",  # Use a smaller buffer to shuffle the chunks
            "-sv": "1",     # Shuffle the videos
            "-gcp": "0"     # Remove parallelism from chunk creation
        },
        # TODO: Adjust the parameters for the normalized ResNet features
        RESNET_NORMALIZED_FEATURE_NAME: {
            "-cds": "0",    # Don't cache the dataset
            "-cpm": "2.0",  # Sample more chunks each time the features are read
            "-cs": "0.08",  # Use a smaller buffer to shuffle the chunks
            "-sv": "1",     # Shuffle the videos
            "-gcp": "0"     # Remove parallelism from chunk creation
        }
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
DELTA_TEST_PARAMETERS = {
    SPLIT_KEY_VALIDATION: {
        "-nmsd": "linear"
    },
    SPLIT_KEY_UNLABELED: {
        "-nmsd": "linear",
        "-tssj": "1"
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
        # reused the hyper-parameters that were found using the Challenge
        # protocol.
        BAIDU_TWO_FEATURE_NAME:
            TRAIN_HYPERPARAMETERS_CHALLENGE_VALIDATED_BAIDU_2,
        RESNET_NORMALIZED_FEATURE_NAME: {
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
    confidence_train_command = _command_spotting_confidence_train(
        specific_features_dir, feature_name, dataset_type, protocol_name,
        run_name, models_dir, labels_dir, splits_dir, base_config_dir,
        memory_setup)
    # Run the confidence model on the validation split, so that the confidence
    # scores can be used during the validation step when training the delta
    # model below.
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
    # Run testing on both the validation (SPLIT_KEY_VALIDATION) and challenge
    # (SPLIT_KEY_UNLABELED) splits.
    last_test_commands = []
    for split_key in [SPLIT_KEY_VALIDATION, SPLIT_KEY_UNLABELED]:
        confidence_and_delta_results_dir = \
            _spotting_confidence_and_delta_results_dir(
                CONFIDENCE, feature_name, split_key, dataset_type,
                protocol_name, run_name, results_dir)
        confidence_test_command = _command_spotting_confidence_test(
            confidence_and_delta_results_dir, specific_features_dir,
            feature_name, split_key, dataset_type, protocol_name, run_name,
            models_dir, labels_dir, splits_dir, base_config_dir)
        delta_test_commands = _commands_spotting_delta_test(
            confidence_and_delta_results_dir, specific_features_dir,
            feature_name, split_key, dataset_type, protocol_name, run_name,
            models_dir, labels_dir, splits_dir, base_config_dir)
        last_test_commands.extend(
            [confidence_test_command] + delta_test_commands)
    return [
        confidence_train_command, confidence_test_validation_command,
        delta_train_command] + last_test_commands


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
    challenge_results_dir = _spotting_confidence_and_delta_results_dir(
        CONFIDENCE, feature_name, SPLIT_KEY_UNLABELED, dataset_type,
        protocol_name, run_name, results_dir)
    confidence_test_command = _command_spotting_confidence_test(
        challenge_results_dir, specific_features_dir, feature_name,
        SPLIT_KEY_UNLABELED, dataset_type, protocol_name, run_name, models_dir,
        labels_dir, splits_dir, base_config_dir)
    delta_test_commands = _commands_spotting_delta_test(
        challenge_results_dir, specific_features_dir, feature_name,
        SPLIT_KEY_UNLABELED, dataset_type, protocol_name, run_name,
        models_dir, labels_dir, splits_dir, base_config_dir)
    return [
        confidence_train_command, delta_train_command,
        confidence_test_command, *delta_test_commands]


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
    dataset_type = DATASET_TYPE_SOCCERNET_V2_CHALLENGE_VALIDATION
    protocol_name = SPOTTING_CHALLENGE_VALIDATED
    # Concatenate the existing Baidu and ResNet confidence scores for both the
    # validation and challenge splits.
    features_from_confidences_commands = [
        _command_features_from_confidence_results(
            split_key, dataset_type, protocol_name, run_name, results_dir,
            features_dir)
        for split_key in [SPLIT_KEY_VALIDATION, SPLIT_KEY_UNLABELED]
    ]
    concatenated_confidence_features_dir = \
        _concatenated_confidence_features_dir(protocol_name, features_dir)
    confidence_averaging_train_command = \
        _command_spotting_confidence_averaging_train(
            concatenated_confidence_features_dir, dataset_type, run_name,
            models_dir, labels_dir, splits_dir, base_config_dir)
    test_commands = []
    for split_key in [SPLIT_KEY_VALIDATION, SPLIT_KEY_UNLABELED]:
        confidence_and_delta_results_dir = \
            _spotting_confidence_and_delta_results_dir(
                AVERAGED_CONFIDENCE, BAIDU_TWO_FEATURE_NAME, split_key,
                dataset_type, protocol_name, run_name, results_dir)
        confidence_averaging_test_command = \
            _command_spotting_confidence_averaging_test(
                confidence_and_delta_results_dir,
                concatenated_confidence_features_dir, split_key, dataset_type,
                run_name, models_dir, labels_dir, splits_dir, base_config_dir)
        # We use the model trained only on the Baidu features when inferring
        # the deltas.
        delta_test_commands = _commands_spotting_delta_test(
            confidence_and_delta_results_dir, baidu_two_features_dir,
            BAIDU_TWO_FEATURE_NAME, split_key, dataset_type, protocol_name,
            run_name, models_dir, labels_dir, splits_dir, base_config_dir)
        test_commands.extend(
            [confidence_averaging_test_command, *delta_test_commands])
    return [
        *features_from_confidences_commands,
        confidence_averaging_train_command, *test_commands]


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
    features_from_confidence_results_command = \
        _command_features_from_confidence_results(
            split_key, dataset_type, protocol_name, run_name, results_dir,
            features_dir)
    confidence_and_delta_results_dir = \
        _spotting_confidence_and_delta_results_dir(
            AVERAGED_CONFIDENCE, BAIDU_TWO_FEATURE_NAME, split_key,
            dataset_type, protocol_name, run_name, results_dir)
    concatenated_confidence_features_dir = \
        _concatenated_confidence_features_dir(protocol_name, features_dir)
    # Confidence averaging uses the existing model from the Challenge Validated
    # protocol, since we don't have a validation set in the Challenge protocol.
    confidence_averaging_test_command = \
        _command_spotting_confidence_averaging_test(
            confidence_and_delta_results_dir,
            concatenated_confidence_features_dir, split_key, dataset_type,
            run_name, models_dir, labels_dir, splits_dir, base_config_dir)
    # We use the model trained only on the Baidu features when inferring
    # the deltas.
    delta_test_commands = _commands_spotting_delta_test(
        confidence_and_delta_results_dir, baidu_two_features_dir,
        BAIDU_TWO_FEATURE_NAME, split_key, dataset_type, protocol_name,
        run_name, models_dir, labels_dir, splits_dir, base_config_dir)
    return [
        features_from_confidence_results_command,
        confidence_averaging_test_command, *delta_test_commands]


# We define print_commands here so that users can import it from the current
# module (spotting_challenge_commands.py).
print_commands = print_command_list


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
    return Command(
        f"Train the confidence model on the {dataset_type} dataset on the "
        f"{feature_name} features",
        SCRIPT_TRAIN,
        {
            "-sd": splits_dir,
            "-ld": labels_dir,
            "-fd": specific_features_dir,
            "-fn": feature_name,
            "-dt": dataset_type,
            "-cd": os.path.join(
                base_config_dir, CONFIG_DIR_CHALLENGE_CONFIDENCE),
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
    model_last_or_best = _model_last_or_best(split_key)
    return Command(
        f"Test the confidence model on the {split_key} split of "
        f"the {dataset_type} dataset on the {feature_name} features",
        SCRIPT_TEST,
        {
            "-sd": splits_dir,
            "-ld": labels_dir,
            "-fd": specific_features_dir,
            "-fn": feature_name,
            "-dt": dataset_type,
            "-cd": os.path.join(
                base_config_dir, CONFIG_DIR_CHALLENGE_CONFIDENCE),
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
    command_arguments = {
        "-sd": splits_dir,
        "-ld": labels_dir,
        "-fd": specific_features_dir,
        "-fn": feature_name,
        "-dt": dataset_type,
        "-cd": os.path.join(base_config_dir, CONFIG_DIR_CHALLENGE_DELTA),
        **MEMORY_TRAIN_PARAMETERS[memory_setup][feature_name],
        **detector_args(DETECTOR_DENSE_DELTA),
        **delta_train_hyperparameters,
        "-m": delta_model_dir,
    }
    if confidence_validation_results_dir:
        command_arguments["-rd"] = confidence_validation_results_dir
    return Command(
        f"Train the displacement (delta) model on the {dataset_type} dataset "
        f"using {feature_name} features",
        SCRIPT_TRAIN,
        command_arguments,
        env_vars=MEMORY_TRAIN_ENVIRONMENT_VARIABLES[memory_setup][feature_name]
    )


def _commands_spotting_delta_test(
        specific_results_dir: str, specific_features_dir: str,
        feature_name: str, split_key: str, dataset_type: str,
        protocol_name: str, run_name: str, models_dir: str, labels_dir: str,
        splits_dir: str, base_config_dir: str) -> List[Command]:
    delta_model_dir = _spotting_delta_model_dir(
        feature_name, dataset_type, protocol_name, run_name, models_dir)
    model_last_or_best = _model_last_or_best(split_key)
    delta_test_command = Command(
        f"Test the displacement (delta) model on the {split_key} split of the "
        f"{dataset_type} dataset using {feature_name} features", SCRIPT_TEST,
        {
            "-sd": splits_dir,
            "-ld": labels_dir,
            "-fd": specific_features_dir,
            "-fn": feature_name,
            "-dt": dataset_type,
            "-cd": os.path.join(
                base_config_dir, CONFIG_DIR_CHALLENGE_DELTA_SOFT_NMS),
            **detector_args(DETECTOR_DENSE_DELTA),
            "-tod": "0",
            **DELTA_TEST_PARAMETERS[split_key],
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
        split_key: str, dataset_type: str, protocol_name: str, run_name: str,
        results_dir: str, features_dir: str) -> Command:
    concatenated_confidence_features_dir = \
        _concatenated_confidence_features_dir(protocol_name, features_dir)
    baidu_two_results_dir = _spotting_confidence_and_delta_results_dir(
        CONFIDENCE, BAIDU_TWO_FEATURE_NAME, split_key, dataset_type,
        protocol_name, run_name, results_dir)
    resnet_normalized_results_dir = _spotting_confidence_and_delta_results_dir(
        CONFIDENCE, RESNET_NORMALIZED_FEATURE_NAME, split_key, dataset_type,
        protocol_name, run_name, results_dir)
    return Command(
        "For each video, read the predicted confidence probabilities from the "
        f"models trained on the Combination x 2 and ResNet features on the"
        f" {split_key} split and concatenate them into a single feature file",
        SCRIPT_CREATE_FEATURES_FROM_RESULTS,
        {
            f"--{FromResultsArgs.INPUT_DIRS}":
                f"{baidu_two_results_dir} {resnet_normalized_results_dir}",
            f"--{FromResultsArgs.OUTPUT_DIR}":
                concatenated_confidence_features_dir,
            f"--{FromResultsArgs.OUTPUT_NAME}": CONFIDENCE
        }
    )


def _command_spotting_confidence_averaging_train(
        concatenated_confidence_features_dir: str, dataset_type: str,
        run_name: str, models_dir: str, labels_dir: str, splits_dir: str,
        base_config_dir: str):
    confidence_averaging_model_file = \
        _spotting_challenge_validated_confidence_averaging_model_file(
            run_name, models_dir)
    return Command(
        "Train a model that averages the confidence probabilities given in the "
        "input features files", SCRIPT_CREATE_AVERAGING_PREDICTOR,
        {
            "-sd": splits_dir,
            "-ld": labels_dir,
            "-fd": concatenated_confidence_features_dir,
            "-fn": CONFIDENCE,
            "-dt": dataset_type,
            "-cd": os.path.join(
                base_config_dir, CONFIG_DIR_CHALLENGE_CONFIDENCE),
            **detector_args(DETECTOR_AVERAGING_CONFIDENCE),
            "-m": confidence_averaging_model_file
        }
    )


def _command_spotting_confidence_averaging_test(
        specific_results_dir: str, concatenated_confidence_features_dir: str,
        split_key: str, dataset_type: str, run_name: str, models_dir: str,
        labels_dir: str, splits_dir: str, base_config_dir: str) -> Command:
    confidence_averaging_model_file = \
        _spotting_challenge_validated_confidence_averaging_model_file(
            run_name, models_dir)
    return Command(
        f"Fuse the confidence scores using the averaging predictor over the "
        f"{split_key} split of the {dataset_type} dataset", SCRIPT_TEST,
        {
            "-sd": splits_dir,
            "-ld": labels_dir,
            "-fd": concatenated_confidence_features_dir,
            "-fn": CONFIDENCE,
            "-dt": dataset_type,
            "-cd": os.path.join(
                base_config_dir, CONFIG_DIR_CHALLENGE_CONFIDENCE),
            **detector_args(DETECTOR_AVERAGING_CONFIDENCE),
            "-ts": split_key,
            "-m": confidence_averaging_model_file,
            "-rd": specific_results_dir,
        }
    )


def _concatenated_confidence_features_dir(
        protocol_name: str, features_dir: str) -> str:
    return os.path.join(
        features_dir, f"{protocol_name}_{CONCATENATED_CONFIDENCE_FEATURES_DIR}")


def _spotting_challenge_validated_confidence_averaging_model_file(
        run_name: str, models_dir: str) -> str:
    averaging_model_file_name = (
        f"{SPOTTING_CHALLENGE_VALIDATED}_{DETECTOR_AVERAGING_CONFIDENCE}"
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
        confidence_name: str, feature_name: str, split_key: str,
        dataset_type: str, protocol_name: str, run_name: str,
        results_dir: str) -> str:
    delta_train_hyperparameters = TRAIN_HYPERPARAMETERS[dataset_type][
        feature_name][DELTA]
    delta_test_parameters = DELTA_TEST_PARAMETERS[split_key]
    delta_all_parameters = {
        **delta_train_hyperparameters,
        **delta_test_parameters}
    extra_name = f"{confidence_name}_{split_key}_{run_name}"
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


def _model_last_or_best(split_key: str) -> str:
    if split_key == SPLIT_KEY_UNLABELED:
        return LAST_MODEL_DIR
    return BEST_MODEL_DIR
