"""Tests for content adapters and an end-to-end pipeline run."""
from content.mixed.adapter import MixedAdapter
from core.pipeline import Pipeline
from core.registry import make_content_adapter, make_task_adapter
from core.router import Router
from core.schema import ContentType, Label, TaskType
from ingestion.base import IterableSource
from labelers.rules_labeler import RulesLabeler


def test_text_adapter_prepare():
    a = make_content_adapter("text")
    item = a.prepare({"id": "1", "text": "  Hello   world  "})
    assert item.payload == "Hello world"
    assert item.features["n_tokens"] == 2
    assert item.content_type == ContentType.TEXT


def test_mixed_adapter_composes_modalities():
    a = MixedAdapter()
    item = a.prepare({"id": "m1",
                      "text": {"text": "hi there"},
                      "image": {"path": "/x.jpg", "width": 100, "height": 50}})
    assert set(item.payload) == {"text", "image"}
    assert item.payload["text"].payload == "hi there"
    assert item.payload["image"].features["aspect_ratio"] == 2.0


def test_pipeline_end_to_end_routes_and_stores():
    content = make_content_adapter("text")
    task = make_task_adapter("single_class", ["spam", "ham"])

    def lf(item):
        return "spam" if "free" in item.payload.lower() else "ham"

    pipe = Pipeline(
        source=IterableSource([
            {"id": "1", "text": "free free free reward"},
            {"id": "2", "text": "lunch tomorrow?"},
        ]),
        content=content,
        task=task,
        labeler=RulesLabeler([lf]),
        router=Router(threshold=0.5),
    )

    reviewed = []

    def review_fn(item, pred):
        reviewed.append(item.id)
        return Label("ham", TaskType.SINGLE_CLASS)

    pipe.run(review_fn=review_fn)
    assert pipe.stats.total == 2
    assert pipe.store.get("1").label.value == "spam"      # single rule -> conf 1.0
    assert pipe.store.get("2").label.value == "ham"
