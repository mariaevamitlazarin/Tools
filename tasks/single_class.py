"""Single-class task: exactly one class label per item."""
from __future__ import annotations

from collections import Counter
from typing import Any

from core.schema import Label, TaskType
from tasks.base import TaskAdapter


class SingleClassTask(TaskAdapter):
    task_type = TaskType.SINGLE_CLASS

    def validate(self, label: Label) -> bool:
        return (
            isinstance(label.value, str)
            and label.value in self._label_space.classes
        )

    def confidence(self, raw_prediction: Any) -> float:
        """Expects raw_prediction = {class: prob, ...}; returns top prob."""
        if isinstance(raw_prediction, dict) and raw_prediction:
            return max(raw_prediction.values())
        return 0.0

    def agreement(self, labels: list[Label]) -> float:
        if not labels:
            return 0.0
        top = Counter(l.value for l in labels).most_common(1)[0][1]
        return top / len(labels)

    def consensus(self, labels: list[Label]) -> Label:
        winner = Counter(l.value for l in labels).most_common(1)[0][0]
        return Label(value=winner, task_type=self.task_type)
