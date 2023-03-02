#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import cProfile
import logging
import pstats

import tensorflow as tf

from spivak.application.argument_parser import get_args
from spivak.application.validation import \
    create_evaluation as create_validation_evaluation

ALLOW_MEMORY_GROWTH = False
TIMING_FILENAME = "validation.prof"


def main():
    pr = cProfile.Profile()
    pr.enable()
    _run_validation()
    pr.disable()
    pr.dump_stats(TIMING_FILENAME)
    p = pstats.Stats(TIMING_FILENAME)
    p.sort_stats('cumulative').print_stats(30)
    import pdb
    pdb.set_trace()


def _run_validation():
    logging.getLogger().setLevel(logging.INFO)
    # disable_eager_execution is used here to match validation.py.
    tf.compat.v1.disable_eager_execution()
    if ALLOW_MEMORY_GROWTH:
        _allow_memory_growth()
    evaluation = create_validation_evaluation(get_args(), best_metric=0.0)
    print(f"Evaluation: {evaluation}")


def _allow_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == "__main__":
    main()
