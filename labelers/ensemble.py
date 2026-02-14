"""Ensemble labeler: combine several labelers into one prediction.

Strategy 'vote' tallies the proposed class values; confidence is the winning
fraction blended with the mean confidence of the winning voters.
"""
from __future__ import annotations

from collections import defaultdict

from core.schema import Item, Label, Prediction
from labelers.base import Labeler
from tasks.base import TaskAdapter


class EnsembleLabeler(Labeler):
    name = "ensemble"

    def __init__(self, labelers: list[Labeler]):
        if not labelers:
            raise ValueError("EnsembleLabeler needs at least one labeler")
        self.labelers = labelers

    def propose(self, item: Item, task: TaskAdapter) -> Prediction:
        preds = [lab.propose(item, task) for lab in self.labelers]

        buckets: dict = defaultdict(list)
        for p in preds:
            key = self._key(p.label.value)
            buckets[key].append(p)

        winner_key = max(buckets, key=lambda k: len(buckets[k]))
        winners = buckets[winner_key]
        vote_frac = len(winners) / len(preds)
        mean_conf = sum(p.confidence for p in winners) / len(winners)
        confidence = 0.5 * vote_frac + 0.5 * mean_conf

        return Prediction(
            item_id=item.id,
            label=Label(value=winners[0].label.value, task_type=task.task_type),
            confidence=confidence,
            labeler_name=self.name,
            raw={"members": [p.labeler_name for p in preds]},
        )

    @staticmethod
    def _key(value):
        if isinstance(value, list):
            return tuple(value)
        return value
