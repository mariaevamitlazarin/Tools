"""Runnable end-to-end demo (offline, stdlib only).

Spam/ham text classification with a rules labeler. Confident items are
auto-accepted; uncertain ones go to a review queue drained by an oracle stub
standing in for a human. Run:  python demo.py
"""
from __future__ import annotations

from core.pipeline import Pipeline
from core.registry import make_content_adapter, make_task_adapter
from core.router import Router
from core.schema import Label
from feedback.collector import FeedbackCollector
from ingestion.base import IterableSource
from labelers.rules_labeler import RulesLabeler

# --- 1. Raw data ------------------------------------------------------------
RAW = [
    {"id": "1", "text": "WIN a FREE prize now, click this link!!!"},
    {"id": "2", "text": "Hey, are we still on for lunch tomorrow?"},
    {"id": "3", "text": "Limited offer, claim your free reward today"},
    {"id": "4", "text": "Project notes attached, see you at standup"},
    {"id": "5", "text": "meeting moved to 3pm free room booked"},  # ambiguous
]

GROUND_TRUTH = {"1": "spam", "2": "ham", "3": "spam", "4": "ham", "5": "ham"}

# --- 2. Weak-supervision labeling functions --------------------------------
SPAM_WORDS = {"win", "free", "prize", "click", "offer", "claim", "reward"}

def lf_spam_words(item):
    vocab = item.features["vocab"]
    return "spam" if len(vocab & SPAM_WORDS) >= 2 else None

def lf_has_link(item):
    return "spam" if "link" in item.payload.lower() else None

def lf_conversational(item):
    convo = {"lunch", "standup", "meeting", "notes", "tomorrow"}
    return "ham" if item.features["vocab"] & convo else None

def lf_free_word(item):
    return "spam" if "free" in item.features["vocab"] else None

# --- 3. Build the pipeline from the registry --------------------------------
content = make_content_adapter("text")
task = make_task_adapter("single_class", classes=["spam", "ham"])
labeler = RulesLabeler([lf_spam_words, lf_has_link, lf_conversational, lf_free_word])

pipe = Pipeline(
    source=IterableSource(RAW),
    content=content,
    task=task,
    labeler=labeler,
    router=Router(threshold=0.9),
)

# --- 4. Human-review stub (oracle uses ground truth) -----------------------
feedback = FeedbackCollector()

def review_fn(item, prediction):
    truth = GROUND_TRUTH[item.id]
    corrected = Label(value=truth, task_type=task.task_type)
    feedback.record(item, prediction, corrected)
    return corrected

# --- 5. Run -----------------------------------------------------------------
pipe.run(review_fn=review_fn)

# --- 6. Report --------------------------------------------------------------
print("Routing:")
print(f"  total      : {pipe.stats.total}")
print(f"  auto       : {pipe.stats.auto}  ({pipe.stats.auto_rate:.0%})")
print(f"  to human   : {pipe.stats.human} ({pipe.stats.human_rate:.0%})")
print(f"\nSource counts: {pipe.store.counts_by_source()}")
print(f"Human change rate (of reviewed): {feedback.change_rate:.0%}")

print("\nFinal labels vs truth:")
for ann in sorted(pipe.store.all(), key=lambda a: a.item_id):
    truth = GROUND_TRUTH[ann.item_id]
    mark = "OK " if ann.label.value == truth else "XX "
    print(f"  {mark} id={ann.item_id} -> {ann.label.value:4s} "
          f"[{ann.source.value}] truth={truth}")

out = pipe.store.export_jsonl("outputs/dataset.jsonl")
print(f"\nExported dataset to {out}")
