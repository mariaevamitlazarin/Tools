"""Trained-classifier labeler.

Wraps any model exposing predict_proba(features) -> {class: prob}. The default
`predict_fn` is a stub; inject your sklearn/torch model's scoring callable.
"""
from __future__ import annotations

from typing import Callable

from core.schema import Item, Label, Prediction
from labelers.base import Labeler
from tasks.base import TaskAdapter

# Maps an Item to a class->probability dict.
PredictFn = Callable[[Item], dict]


class ClassifierLabeler(Labeler):
    name = "classifier"

    def __init__(self, predict_fn: PredictFn):
        self.predict_fn = predict_fn

    def propose(self, item: Item, task: TaskAdapter) -> Prediction:
        probs = self.predict_fn(item)
        top = max(probs, key=probs.get)
        label = Label(value=top, task_type=task.task_type)
        return Prediction(
            item_id=item.id, label=label,
            confidence=task.confidence(probs),
            labeler_name=self.name, raw=probs,
        )
