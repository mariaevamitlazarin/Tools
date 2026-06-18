"""Routing logic: decide whether a prediction is trustworthy enough to accept
automatically, or should be sent to a human review queue.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.schema import Prediction
from tasks.base import TaskAdapter


@dataclass
class Router:
    threshold: float = 0.85   # accept automatically at or above this confidence

    def is_confident(self, prediction: Prediction, task: TaskAdapter) -> bool:
        if not task.validate(prediction.label):
            return False      # malformed labels always go to a human
        return prediction.confidence >= self.threshold

    def route(self, prediction: Prediction, task: TaskAdapter) -> str:
        return "auto" if self.is_confident(prediction, task) else "human"
