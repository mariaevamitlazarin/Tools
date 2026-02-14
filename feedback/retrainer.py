"""Retrainer: close the loop using collected corrections.

Stub interface. For a classifier, `fit` would retrain on training_pairs; for an
LLM labeler, it would append high-signal corrections as few-shot examples.
"""
from __future__ import annotations

from core.schema import Item, Label


class Retrainer:
    def fit(self, pairs: list[tuple[Item, Label]]) -> None:
        # Real impl: featurize items, train/fine-tune, persist model artifact.
        raise NotImplementedError("Wire in your training routine here.")

    def build_few_shot(self, pairs: list[tuple[Item, Label]], k: int = 5) -> str:
        """Turn corrections into a few-shot block for an LLM labeler."""
        lines = []
        for item, label in pairs[:k]:
            content = item.payload if isinstance(item.payload, str) else str(item.payload)
            lines.append(f"Content: {content}\nLabel: {label.value}")
        return "\n\n".join(lines)
