"""Pipeline metrics: routing efficiency and annotation quality."""
from __future__ import annotations

from dataclasses import dataclass

from core.schema import Label
from tasks.base import TaskAdapter


@dataclass
class RoutingStats:
    total: int = 0
    auto: int = 0
    human: int = 0

    def record(self, decision: str) -> None:
        self.total += 1
        if decision == "auto":
            self.auto += 1
        else:
            self.human += 1

    @property
    def auto_rate(self) -> float:
        return self.auto / self.total if self.total else 0.0

    @property
    def human_rate(self) -> float:
        return self.human / self.total if self.total else 0.0


def auto_accept_precision(
    auto_labels: dict[str, Label],
    gold_labels: dict[str, Label],
    task: TaskAdapter,
) -> float:
    """Of items auto-accepted that also have gold truth, fraction that agree."""
    overlap = [k for k in auto_labels if k in gold_labels]
    if not overlap:
        return 0.0
    correct = sum(
        task.agreement([auto_labels[k], gold_labels[k]]) >= 1.0 for k in overlap
    )
    return correct / len(overlap)


def mean_agreement(
    per_item_labels: dict[str, list[Label]], task: TaskAdapter
) -> float:
    """Average inter-annotator agreement across items with >= 2 labels."""
    scored = [
        task.agreement(labels)
        for labels in per_item_labels.values()
        if len(labels) >= 2
    ]
    return sum(scored) / len(scored) if scored else 0.0
