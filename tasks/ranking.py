"""Ranking task: an ordered list of item/option IDs."""
from __future__ import annotations

from itertools import combinations
from typing import Any

from core.schema import Label, TaskType
from tasks.base import TaskAdapter


class RankingTask(TaskAdapter):
    task_type = TaskType.RANKING

    def validate(self, label: Label) -> bool:
        if not isinstance(label.value, list):
            return False
        allowed = set(self._label_space.classes)
        return (
            set(label.value) <= allowed
            and len(set(label.value)) == len(label.value)   # no duplicates
        )

    def confidence(self, raw_prediction: Any) -> float:
        """Expects {"scores": [...]} of per-position margins; mean margin."""
        scores = (raw_prediction or {}).get("scores") if isinstance(raw_prediction, dict) else None
        return (sum(scores) / len(scores)) if scores else 0.0

    def agreement(self, labels: list[Label]) -> float:
        """Mean pairwise Kendall-tau-style concordance over shared items."""
        if len(labels) < 2:
            return 1.0
        sims, pairs = 0.0, 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sims += self._concordance(labels[i].value, labels[j].value)
                pairs += 1
        return sims / pairs if pairs else 1.0

    @staticmethod
    def _concordance(a: list, b: list) -> float:
        common = [x for x in a if x in b]
        if len(common) < 2:
            return 1.0
        rank_a = {x: a.index(x) for x in common}
        rank_b = {x: b.index(x) for x in common}
        conc = disc = 0
        for x, y in combinations(common, 2):
            sa = rank_a[x] - rank_a[y]
            sb = rank_b[x] - rank_b[y]
            if sa * sb > 0:
                conc += 1
            else:
                disc += 1
        total = conc + disc
        return conc / total if total else 1.0

    def consensus(self, labels: list[Label]) -> Label:
        """Borda count: lower mean position wins."""
        from collections import defaultdict
        pos_sum: dict[Any, float] = defaultdict(float)
        seen: dict[Any, int] = defaultdict(int)
        for l in labels:
            for idx, item in enumerate(l.value):
                pos_sum[item] += idx
                seen[item] += 1
        order = sorted(pos_sum, key=lambda x: pos_sum[x] / seen[x])
        return Label(value=order, task_type=self.task_type)
