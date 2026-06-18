"""Collect human corrections so they can feed retraining / prompt refinement."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import Item, Label, Prediction


@dataclass
class Correction:
    item: Item
    proposed: Label
    corrected: Label
    was_changed: bool


@dataclass
class FeedbackCollector:
    corrections: list[Correction] = field(default_factory=list)

    def record(self, item: Item, prediction: Prediction, corrected: Label) -> Correction:
        changed = prediction.label.value != corrected.value
        c = Correction(item, prediction.label, corrected, changed)
        self.corrections.append(c)
        return c

    @property
    def change_rate(self) -> float:
        """Fraction of reviewed items the human actually altered."""
        if not self.corrections:
            return 0.0
        return sum(c.was_changed for c in self.corrections) / len(self.corrections)

    def training_pairs(self) -> list[tuple[Item, Label]]:
        """(item, corrected_label) pairs for retraining."""
        return [(c.item, c.corrected) for c in self.corrections]
