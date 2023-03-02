# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.
#
# This file incorporates work covered by the following copyright and permission
# notice:
#   Copyright (c) 2021 Jannes Elstner
#   Licensed under the terms of the MIT license.
#   You may obtain a copy of the MIT License at https://opensource.org/licenses/MIT

# This file contains pieces of code taken from the following file.
# https://github.com/Jannoshh/simple-sam/blob/cd77d0217128aa4ad3dcea558403b5bca93b5952/sam.py
# At Yahoo Inc., the code was modified and new code was added, in order to
# address our specific use-case.
#
# Relevant paper:
# Sharpness-Aware Minimization for Efficiently Improving Generalization
# https://arxiv.org/pdf/2010.01412.pdf

from types import MethodType

import tensorflow as tf
from tensorflow.keras import Model


def maybe_convert_model_to_sam(model: Model, rho: float, eps: float) -> None:
    if rho > 0.0:
        # Monkey-patch the models object, since I wasn't able to load it to/from
        # disk using Keras due to some complications with the Model and
        # Functional classes.
        model._rho = rho
        model._eps = eps
        model.train_step = MethodType(SAMModel.train_step, model)


class SAMModel(Model):

    def __init__(
            self, main_input, output_tensors, rho: float, eps: float) -> None:
        super(SAMModel, self).__init__(main_input, output_tensors)
        self._rho = rho
        self._eps = eps

    def train_step(self, data):
        # Unpack the data. Its structure depends on your models and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                y, y_pred, sample_weight=sample_weight,
                regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # first step
        e_ws = []
        grad_norm = tf.linalg.global_norm(gradients)
        for i in range(len(trainable_vars)):
            if gradients[i] is not None:
                e_w = tf.math.scalar_mul(self._rho, gradients[i]) / (
                        grad_norm + self._eps)
            else:
                e_w = tf.math.scalar_mul(0.0, trainable_vars[i])
            trainable_vars[i].assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                y, y_pred, sample_weight=sample_weight,
                regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        for i in range(len(trainable_vars)):
            trainable_vars[i].assign_sub(e_ws[i])
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(
            y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
