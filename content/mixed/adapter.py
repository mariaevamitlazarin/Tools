"""Mixed content adapter.

Composes per-modality adapters. A raw record is a dict whose keys map to
modalities, e.g. {"id": ..., "text": {...}, "image": {...}}. Each present
modality is prepared by its own adapter and stored under item.payload[modality].
"""
from __future__ import annotations

from typing import Any

from content.audio.adapter import AudioAdapter
from content.base import ContentAdapter
from content.image.adapter import ImageAdapter
from content.text.adapter import TextAdapter
from core.schema import ContentType, Item


class MixedAdapter(ContentAdapter):
    content_type = ContentType.MIXED

    def __init__(self):
        self._adapters = {
            "text": TextAdapter(),
            "image": ImageAdapter(),
            "audio": AudioAdapter(),
        }

    def load(self, raw: Any) -> Item:
        if not isinstance(raw, dict):
            raise TypeError("MixedAdapter expects a dict record")
        item_id = str(raw.get("id", "mixed"))
        present = {m: raw[m] for m in self._adapters if m in raw}
        return Item(id=item_id, content_type=self.content_type,
                    payload=present, metadata={"modalities": list(present)})

    def preprocess(self, item: Item) -> Item:
        prepared = {}
        for modality, sub_raw in item.payload.items():
            prepared[modality] = self._adapters[modality].prepare(sub_raw)
        item.payload = prepared          # payload[modality] -> sub-Item
        return item

    def featurize(self, item: Item) -> Item:
        item.features = {
            m: sub.features for m, sub in item.payload.items()
        }
        return item
