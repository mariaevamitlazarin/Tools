"""Canonical data structures shared across every stage of the pipeline.

These are deliberately content- and task-agnostic. Content adapters and task
adapters interpret the generic fields (`payload`, `value`) in their own way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MIXED = "mixed"


class TaskType(str, Enum):
    SINGLE_CLASS = "single_class"
    MULTI_LABEL = "multi_label"
    SPANS = "spans"          # text spans
    BBOX = "bbox"            # image bounding boxes
    RANKING = "ranking"


class LabelSource(str, Enum):
    AUTO = "auto"            # accepted automatically by a labeler
    HUMAN = "human"          # confirmed/corrected by an annotator
    GOLD = "gold"            # known-truth seed item for QC


@dataclass
class Item:
    """A single unit of content flowing through the pipeline."""
    id: str
    content_type: ContentType
    payload: Any                              # raw or preprocessed content
    metadata: dict[str, Any] = field(default_factory=dict)
    features: Optional[Any] = None            # embeddings / derived features


@dataclass
class Label:
    """A task-specific label. `value` shape depends on the TaskAdapter:

      single_class -> str
      multi_label  -> list[str]
      spans        -> list[dict(start, end, tag)]
      bbox         -> list[dict(x, y, w, h, tag)]
      ranking      -> list[str]  (ordered)
    """
    value: Any
    task_type: TaskType


@dataclass
class Prediction:
    """A labeler's proposal for an item, with a confidence in [0, 1]."""
    item_id: str
    label: Label
    confidence: float
    labeler_name: str
    raw: Any = None                           # raw model output, for debugging


@dataclass
class Annotation:
    """A finalized record: the label plus provenance for auditing."""
    item_id: str
    label: Label
    source: LabelSource
    confidence: Optional[float] = None
    annotator_id: Optional[str] = None
    created_at: datetime = field(default_factory=_now)
    notes: Optional[str] = None


@dataclass
class LabelSpace:
    """Describes the legal set of labels for a task instance."""
    task_type: TaskType
    classes: list[str] = field(default_factory=list)   # for class/tag-based tasks
    extra: dict[str, Any] = field(default_factory=dict)
