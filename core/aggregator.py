"""Aggregator: resolve multiple labels per item into a single consensus label."""
from __future__ import annotations

from collections import defaultdict

from core.schema import Label
from tasks.base import TaskAdapter


class Aggregator:
    def __init__(self, task: TaskAdapter):
        self.task = task
        self._pending: dict[str, list[Label]] = defaultdict(list)

    def add(self, item_id: str, label: Label) -> None:
        self._pending[item_id].append(label)

    def finalize(self) -> dict[str, Label]:
        """Collapse each item's labels into one consensus label."""
        out: dict[str, Label] = {}
        for item_id, labels in self._pending.items():
            out[item_id] = (
                labels[0] if len(labels) == 1 else self.task.consensus(labels)
            )
        return out
