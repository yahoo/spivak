# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

# Options for memory usage setups.
MEMORY_SETUP_256GB = "256"
MEMORY_SETUP_64GB = "64"
# Change the constants below according to your local setup.
FEATURES_DIR = "./data/features/"
BAIDU_FEATURES_DIR = "./data/features/baidu/"
BAIDU_TWO_FEATURES_DIR = "./data/features/baidu_2.0/"
RESNET_FEATURES_DIR = "./data/features/resnet/"
RESNET_NORMALIZED_FEATURES_DIR = "./data/features/resnet_normalized/"
LABELS_DIR = "./data/labels/"
SPLITS_DIR = "./data/splits/"
BASE_CONFIG_DIR = "./configs/"
MODELS_DIR = "YOUR_MODELS_DIR"
RESULTS_DIR = "YOUR_RESULTS_DIR"
RUN_NAME = "first"
MEMORY_SETUP = MEMORY_SETUP_256GB

# You might have your data set up in such a way that all the SoccerNet features
# and labels are under the same directory tree. In that case, each game folder
# would contain its respective Baidu (Combination) features, ResNet features,
# and action spotting labels. For example, if all your data is in a base
# directory called "all_soccernet_features_and_labels/", then a game folder
# might have the following files:
#
# $ ls all_soccernet_features_and_labels/england_epl/2014-2015/2015-02-21\ -\ 18-00\ Chelsea\ 1\ -\ 1\ Burnley/
#   1_baidu_soccer_embeddings.npy  2_baidu_soccer_embeddings.npy
#   1_ResNET_TF2.npy  1_ResNET_TF2_PCA512.npy  2_ResNET_TF2.npy
#   2_ResNET_TF2_PCA512.npy  Labels-cameras.json  Labels-v2.json
#
# In that case, you could just set some particular constants above to the same
# base directory, as follows:
# BAIDU_FEATURES_DIR = "all_soccernet_features_and_labels/"
# BAIDU_TWO_FEATURES_DIR = "all_soccernet_features_and_labels/"
# RESNET_FEATURES_DIR = "all_soccernet_features_and_labels/"
# RESNET_NORMALIZED_FEATURES_DIR = "all_soccernet_features_and_labels/"
# LABELS_DIR = "all_soccernet_features_and_labels/"
#
# We suggest setting FEATURES_DIR to a different folder, like
# "./data/features/". At the same time, you probably won't have to change
# SPLITS_DIR and BASE_CONFIG_DIR.
