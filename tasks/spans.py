"""Span task (text): label regions as list of {start, end, tag}."""
from __future__ import annotations

from typing import Any

from core.schema import Label, TaskType
from tasks.base import TaskAdapter


def _overlap(a: dict, b: dict) -> int:
    lo, hi = max(a["start"], b["start"]), min(a["end"], b["end"])
    return max(0, hi - lo)


class SpansTask(TaskAdapter):
    task_type = TaskType.SPANS

    def validate(self, label: Label) -> bool:
        if not isinstance(label.value, list):
            return False
        allowed = set(self._label_space.classes)
        for s in label.value:
            if not isinstance(s, dict):
                return False
            if not {"start", "end", "tag"} <= s.keys():
                return False
            if s["start"] >= s["end"] or s["tag"] not in allowed:
                return False
        return True

    def confidence(self, raw_prediction: Any) -> float:
        """Expects {"spans": [...], "scores": [...]}; mean span score."""
        scores = (raw_prediction or {}).get("scores") if isinstance(raw_prediction, dict) else None
        return (sum(scores) / len(scores)) if scores else 0.0

    def agreement(self, labels: list[Label]) -> float:
        """Mean pairwise span-overlap IoU with matching tags."""
        if len(labels) < 2:
            return 1.0
        sims, pairs = 0.0, 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sims += self._iou(labels[i].value, labels[j].value)
                pairs += 1
        return sims / pairs if pairs else 1.0

    @staticmethod
    def _iou(a_spans: list[dict], b_spans: list[dict]) -> float:
        inter = total = 0
        for a in a_spans:
            for b in b_spans:
                if a["tag"] == b["tag"]:
                    inter += _overlap(a, b)
        a_len = sum(s["end"] - s["start"] for s in a_spans)
        b_len = sum(s["end"] - s["start"] for s in b_spans)
        total = a_len + b_len - inter
        return inter / total if total else 1.0

    def consensus(self, labels: list[Label]) -> Label:
        """Union of spans; production systems should merge overlaps per tag."""
        merged = [s for l in labels for s in l.value]
        return Label(value=merged, task_type=self.task_type)
