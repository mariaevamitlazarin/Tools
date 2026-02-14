"""Human-in-the-loop review queue.

Items land here when the router is not confident. They are ordered by review
priority (least-confident first) so annotator effort goes where it matters most.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Optional

from core.active_learning import margin_priority
from core.schema import Item, Prediction


@dataclass(order=True)
class _Entry:
    priority: float
    seq: int
    item: Item = field(compare=False)
    prediction: Prediction = field(compare=False)


class ReviewQueue:
    def __init__(self):
        self._heap: list[_Entry] = []
        self._counter = itertools.count()

    def push(self, item: Item, prediction: Prediction) -> None:
        entry = _Entry(
            priority=margin_priority(prediction),
            seq=next(self._counter),
            item=item,
            prediction=prediction,
        )
        heapq.heappush(self._heap, entry)

    def pop(self) -> Optional[tuple[Item, Prediction]]:
        if not self._heap:
            return None
        entry = heapq.heappop(self._heap)
        return entry.item, entry.prediction

    def __len__(self) -> int:
        return len(self._heap)
