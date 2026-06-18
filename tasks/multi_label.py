"""Multi-label task: a set of class labels per item."""
from __future__ import annotations

from collections import Counter
from typing import Any

from core.schema import Label, TaskType
from tasks.base import TaskAdapter


class MultiLabelTask(TaskAdapter):
    task_type = TaskType.MULTI_LABEL

    def validate(self, label: Label) -> bool:
        if not isinstance(label.value, (list, set, tuple)):
            return False
        allowed = set(self._label_space.classes)
        return all(v in allowed for v in label.value)

    def confidence(self, raw_prediction: Any) -> float:
        """Expects {class: prob}; confidence = mean over labels above 0.5,
        else 1 - max (model is confident nothing applies)."""
        if not isinstance(raw_prediction, dict) or not raw_prediction:
            return 0.0
        positives = [p for p in raw_prediction.values() if p >= 0.5]
        if positives:
            return sum(positives) / len(positives)
        return 1.0 - max(raw_prediction.values())

    def agreement(self, labels: list[Label]) -> float:
        """Mean pairwise Jaccard similarity across label sets."""
        sets = [set(l.value) for l in labels]
        if len(sets) < 2:
            return 1.0
        sims, pairs = 0.0, 0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                sims += (len(sets[i] & sets[j]) / len(union)) if union else 1.0
                pairs += 1
        return sims / pairs if pairs else 1.0

    def consensus(self, labels: list[Label]) -> Label:
        """Keep classes endorsed by a majority (more than half) of annotators.
        For an even split the label is dropped, since there is no majority."""
        counts = Counter(v for l in labels for v in l.value)
        threshold = len(labels) / 2
        agreed = sorted(c for c, n in counts.items() if n > threshold)
        return Label(value=agreed, task_type=self.task_type)
