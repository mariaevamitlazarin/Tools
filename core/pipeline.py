"""End-to-end orchestrator.

Composes a content adapter, a task adapter, a labeler, a router, a review queue
and a store. The orchestrator is agnostic to modality and task — it only speaks
the abstract interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from content.base import ContentAdapter
from core.metrics import RoutingStats
from core.router import Router
from core.schema import Item, LabelSource, Prediction
from core.store import DatasetStore
from ingestion.base import IngestionSource
from labelers.base import Labeler
from review.queue import ReviewQueue
from tasks.base import TaskAdapter

# A human-review callback: given (item, prediction) returns the confirmed Label.
# In production this is the review UI; in tests it can be a stub/oracle.
ReviewFn = Callable[[Item, Prediction], "object"]


@dataclass
class Pipeline:
    source: IngestionSource
    content: ContentAdapter
    task: TaskAdapter
    labeler: Labeler
    router: Router = field(default_factory=Router)
    store: DatasetStore = field(default_factory=DatasetStore)
    review_queue: ReviewQueue = field(default_factory=ReviewQueue)
    stats: RoutingStats = field(default_factory=RoutingStats)

    def run_auto(self) -> "Pipeline":
        """Stage 1: ingest, label, and route. Confident items are accepted;
        uncertain ones are pushed to the review queue."""
        for raw in self.source.stream():
            item = self.content.prepare(raw)
            pred: Prediction = self.labeler.propose(item, self.task)
            decision = self.router.route(pred, self.task)
            self.stats.record(decision)
            if decision == "auto":
                self.store.accept(
                    item.id, pred.label, LabelSource.AUTO, pred.confidence,
                )
            else:
                self.review_queue.push(item, pred)
        return self

    def run_review(self, review_fn: ReviewFn, annotator_id: str = "human") -> "Pipeline":
        """Stage 2: drain the review queue through a human (or oracle stub)."""
        while True:
            nxt = self.review_queue.pop()
            if nxt is None:
                break
            item, pred = nxt
            confirmed = review_fn(item, pred)
            self.store.accept(
                item.id, confirmed, LabelSource.HUMAN,
                confidence=None, annotator_id=annotator_id,
            )
        return self

    def run(self, review_fn: Optional[ReviewFn] = None) -> "Pipeline":
        self.run_auto()
        if review_fn is not None:
            self.run_review(review_fn)
        return self
