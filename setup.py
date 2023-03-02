# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import os
import glob
from setuptools import setup, find_namespace_packages

SCRIPTS_DIR = "bin"
PACKAGE_DIR = "spivak"
# TODO: make the version requirements into ranges.
INSTALL_REQUIRES = [
    "numpy~=1.18.5",
    "scipy~=1.4.1",
    "scikit-learn~=0.24.2",
    "pandas~=1.1.5",
    "Pillow~=8.4.0",
    "opencv-python~=4.5.4.58",
    "tqdm~=4.62.3",
    # Note: some portion of the code here does not depend on TensorFlow,
    # so if you are not interested in using the models, you probably
    # don't really have to install TensorFlow.
    "tensorflow~=2.3.0",
    # This version of tensorflow_probability should be compatible with
    # Tensorflow 2.3.
    "tensorflow_probability~=0.11.1",
    # tensorflow-addons has the decoupled weight decay functionality.
    # For TensorFlow 2.3, we need to use this older version (0.13) of the
    # addons package.
    "tensorflow-addons~=0.13.0",
    "plotly~=5.4.0",
    # kaleido is used for being able to render pdf plots with plotly.
    "kaleido~=0.2.1",
    # scikit-video (skvideo), moviepy, imutils are only used in
    # spivak.feature_extraction.SoccerNetDataLoader.py.
    "scikit-video~=1.1.11",
    "moviepy~=1.0.3",
    "imutils~=0.5.4",
    # av is used only in bin/create_visualizations.py and inside
    # spivak.video_visualization, for creating videos with visualizations of
    # results.
    "av~=10.0.0",
    # packaging is only used by
    # spivak.models.assembly.huggingface_activations.py.
    "packaging~=21.2"
]
packages = find_namespace_packages(include=[f"{PACKAGE_DIR}.*"])
scripts = glob.glob(os.path.join(SCRIPTS_DIR, "*.py"))
setup(
    name=PACKAGE_DIR,
    packages=packages,
    install_requires=INSTALL_REQUIRES,
    scripts=scripts,
)
