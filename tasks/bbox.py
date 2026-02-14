"""Bounding-box task (image): list of {x, y, w, h, tag}."""
from __future__ import annotations

from typing import Any

from core.schema import Label, TaskType
from tasks.base import TaskAdapter


def _box_iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0


class BBoxTask(TaskAdapter):
    task_type = TaskType.BBOX

    def validate(self, label: Label) -> bool:
        if not isinstance(label.value, list):
            return False
        allowed = set(self._label_space.classes)
        for box in label.value:
            if not isinstance(box, dict):
                return False
            if not {"x", "y", "w", "h", "tag"} <= box.keys():
                return False
            if box["w"] <= 0 or box["h"] <= 0 or box["tag"] not in allowed:
                return False
        return True

    def confidence(self, raw_prediction: Any) -> float:
        scores = (raw_prediction or {}).get("scores") if isinstance(raw_prediction, dict) else None
        return (sum(scores) / len(scores)) if scores else 0.0

    def agreement(self, labels: list[Label]) -> float:
        if len(labels) < 2:
            return 1.0
        sims, pairs = 0.0, 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sims += self._match_iou(labels[i].value, labels[j].value)
                pairs += 1
        return sims / pairs if pairs else 1.0

    @staticmethod
    def _match_iou(a_boxes: list[dict], b_boxes: list[dict]) -> float:
        if not a_boxes and not b_boxes:
            return 1.0
        if not a_boxes or not b_boxes:
            return 0.0
        best = []
        for a in a_boxes:
            scores = [_box_iou(a, b) for b in b_boxes if a["tag"] == b["tag"]]
            best.append(max(scores) if scores else 0.0)
        return sum(best) / len(best)

    def consensus(self, labels: list[Label]) -> Label:
        merged = [b for l in labels for b in l.value]
        return Label(value=merged, task_type=self.task_type)
