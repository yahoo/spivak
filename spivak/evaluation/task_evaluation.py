# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from abc import ABCMeta, abstractmethod
from typing import Dict


class TaskEvaluation(metaclass=ABCMeta):

	"""Defines the evaluation results for any given single task."""

	@abstractmethod
	def scalars_for_logging(self) -> Dict[str, float]:
		pass

	@abstractmethod
	def summary(self) -> str:
		pass
