"""Base class for labeling-task adapters.

A TaskAdapter defines the label space, validates labels, computes confidence
from a raw prediction, and measures inter-annotator agreement. It knows nothing
about content modality.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.schema import Label, LabelSpace, TaskType


class TaskAdapter(ABC):
    task_type: TaskType

    def __init__(self, label_space: LabelSpace):
        self._label_space = label_space

    def label_space(self) -> LabelSpace:
        return self._label_space

    @abstractmethod
    def validate(self, label: Label) -> bool:
        """Return True if `label.value` is well-formed for this task."""

    @abstractmethod
    def confidence(self, raw_prediction: Any) -> float:
        """Derive a scalar confidence in [0, 1] from a labeler's raw output."""

    @abstractmethod
    def agreement(self, labels: list[Label]) -> float:
        """Agreement score in [0, 1] across labels for the same item."""

    @abstractmethod
    def consensus(self, labels: list[Label]) -> Label:
        """Merge multiple labels for one item into a single agreed label."""
