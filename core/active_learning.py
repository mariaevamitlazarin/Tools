"""Active learning: prioritize the most informative items for human review."""
from __future__ import annotations

from core.schema import Prediction


def uncertainty_sampling(
    predictions: list[Prediction], k: int
) -> list[Prediction]:
    """Return the k predictions the model is least confident about."""
    return sorted(predictions, key=lambda p: p.confidence)[:k]


def margin_priority(prediction: Prediction) -> float:
    """Lower score = higher review priority. Distance below full certainty."""
    return 1.0 - prediction.confidence
