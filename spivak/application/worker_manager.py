# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import logging
import warnings
from multiprocessing import Queue, Process
from typing import Callable, Optional, Tuple

MODULE_ADDONS_INSTALL = "tensorflow_addons.utils.ensure_tf_install"


class Manager:

    def __init__(self, process, input_queue, output_queue):
        self.process = process
        self.input_queue = input_queue
        self.output_queue = output_queue


class ChildTask:

    def __init__(self, do_exit: bool, args: Optional[Tuple]) -> None:
        self.do_exit = do_exit
        self.args = args


def manager_function(
        input_queue: Queue, output_queue: Queue,
        worker_function: Callable) -> None:
    # I tried also doing this with a pool (setting maxtasksperchild to 1),
    # but for some unknown reason it would sometimes not work (maybe a
    # deadlock), but I didn't investigate to understand why.
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings(
        action="ignore", category=UserWarning, module=MODULE_ADDONS_INSTALL)
    logging.info("MANAGER: initializing")
    worker_result_queue = Queue()

    def worker_function_with_queue(*worker_args) -> None:
        result = worker_function(*worker_args)
        worker_result_queue.put(result)

    do_exit = False
    while not do_exit:
        logging.info("MANAGER: waiting for a task")
        child_task = input_queue.get()
        do_exit = child_task.do_exit
        logging.info("MANAGER: Got a task")
        if not child_task.do_exit:
            child = Process(
                target=worker_function_with_queue, args=child_task.args)
            child.start()
            logging.info("MANAGER: Waiting for result")
            output_queue.put(worker_result_queue.get())
            child.join()
    logging.info("MANAGER: done, reached end of function.")
