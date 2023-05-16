#!/usr/bin/env python3
# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from multiprocessing import Process, Queue

from spivak.application.argument_parser import get_args, SharedArgs
from spivak.application.worker_manager import manager_function, Manager


def main() -> None:
    # Due to a memory leak in Keras, we compute our custom validation metric
    # (average-mAP) in a separate process. In contrast, the validation loss is
    # still computed within Keras, which might leak some memory as well,
    # but is manageable. Some people had luck getting around similar memory
    # leaks by either using tf.compat.v1.disable_eager_execution(), changing
    # the threading configuration, or running garbage collection every once in
    # a while, but those options didn't work out for me. See:
    # https://stackoverflow.com/questions/58137677/keras-model-training-memory-leak/58138230#58138230
    # https://github.com/tensorflow/tensorflow/issues/22098
    #
    # Additionally, another memory leak seems to be mitigated by running in
    # eager mode, which also allows us to clear the keras session while
    # running training to keep the memory usage low (which is only possible in
    # eager mode). In theory, eager mode is slower, but I didn't notice much
    # of a speed difference, so am using it. See:
    # https://github.com/tensorflow/tensorflow/issues/31312
    manager = _create_manager()
    manager.process.start()
    _import_and_train(get_args(), manager)
    manager.process.join()


def _create_manager() -> Manager:
    input_queue = Queue()
    output_queue = Queue()
    manager_process = Process(
        target=manager_function, args=(
            input_queue, output_queue, _import_and_compute_validation_result))
    return Manager(manager_process, input_queue, output_queue)


def _import_and_train(args: SharedArgs, manager: Manager) -> None:
    from spivak.application.train_utils import train
    train(args, manager)


def _import_and_compute_validation_result(args, best_metric, epoch):
    from spivak.application.validation import compute_validation_result
    return compute_validation_result(args, best_metric, epoch)


if __name__ == "__main__":
    main()
