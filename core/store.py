"""Dataset store: holds finalized annotations and exports versioned datasets.

This reference implementation is in-memory with JSONL export. Swap the body for
a database or object store without changing callers.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from core.schema import Annotation, Label, LabelSource


def _encode(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _encode(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


class DatasetStore:
    def __init__(self):
        self._annotations: dict[str, Annotation] = {}

    def accept(
        self,
        item_id: str,
        label: Label,
        source: LabelSource,
        confidence: Optional[float] = None,
        annotator_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Annotation:
        ann = Annotation(
            item_id=item_id,
            label=label,
            source=source,
            confidence=confidence,
            annotator_id=annotator_id,
            notes=notes,
        )
        self._annotations[item_id] = ann   # last-write-wins; human overrides auto
        return ann

    def get(self, item_id: str) -> Optional[Annotation]:
        return self._annotations.get(item_id)

    def all(self) -> list[Annotation]:
        return list(self._annotations.values())

    def counts_by_source(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self._annotations.values():
            out[a.source.value] = out.get(a.source.value, 0) + 1
        return out

    def export_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for ann in self._annotations.values():
                f.write(json.dumps(_encode(ann), ensure_ascii=False) + "\n")
        return path
