#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging

from spivak.application.argument_parser import get_args
from spivak.application.test_utils import test


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    test(get_args())
