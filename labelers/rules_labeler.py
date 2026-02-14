"""Rules / weak-supervision labeler.

Runs entirely offline. You provide labeling functions; each returns a partial
vote. Votes are tallied into a pseudo-probability distribution. Useful as a
baseline and for bootstrapping before a trained model exists.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

from core.schema import Item, Label, Prediction
from labelers.base import Labeler
from tasks.base import TaskAdapter

# A labeling function inspects an Item and optionally votes for a class string.
LabelingFunction = Callable[[Item], "str | None"]


class RulesLabeler(Labeler):
    name = "rules"

    def __init__(self, functions: list[LabelingFunction]):
        self.functions = functions

    def propose(self, item: Item, task: TaskAdapter) -> Prediction:
        votes: dict[str, int] = defaultdict(int)
        for fn in self.functions:
            v = fn(item)
            if v is not None:
                votes[v] += 1

        classes = task.label_space().classes
        total = sum(votes.values())
        if total == 0:
            # Abstain -> uniform, lowest confidence.
            probs = {c: 1.0 / len(classes) for c in classes}
        else:
            probs = {c: votes.get(c, 0) / total for c in classes}

        top = max(probs, key=probs.get)
        label = Label(value=top, task_type=task.task_type)
        confidence = task.confidence(probs)
        return Prediction(
            item_id=item.id, label=label, confidence=confidence,
            labeler_name=self.name, raw=probs,
        )
