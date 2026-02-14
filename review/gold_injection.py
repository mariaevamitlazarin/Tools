"""Gold injection: seed the review queue with known-truth items to measure
annotator accuracy. Annotators don't know which items are gold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import Item, Label
from tasks.base import TaskAdapter


@dataclass
class GoldTracker:
    task: TaskAdapter
    truth: dict[str, Label] = field(default_factory=dict)
    results: list[bool] = field(default_factory=list)

    def register(self, item: Item, true_label: Label) -> None:
        self.truth[item.id] = true_label

    def is_gold(self, item_id: str) -> bool:
        return item_id in self.truth

    def check(self, item_id: str, annotator_label: Label) -> bool:
        """Record whether an annotator's label matches gold (agreement == 1)."""
        truth = self.truth[item_id]
        correct = self.task.agreement([annotator_label, truth]) >= 1.0
        self.results.append(correct)
        return correct

    @property
    def accuracy(self) -> float:
        return sum(self.results) / len(self.results) if self.results else 0.0
