"""Image content adapter.

`load` accepts a dict {"id", "path"/"bytes", "width", "height", ...}.
Decoding/resizing is stubbed (no Pillow dependency); wire in real decoding and
a vision embedding model in `preprocess`/`featurize` for production.
"""
from __future__ import annotations

from typing import Any

from content.base import ContentAdapter
from core.schema import ContentType, Item


class ImageAdapter(ContentAdapter):
    content_type = ContentType.IMAGE

    def load(self, raw: Any) -> Item:
        if not isinstance(raw, dict):
            raise TypeError("ImageAdapter expects a dict record")
        item_id = str(raw.get("id") or raw.get("path", "image"))
        payload = {
            "path": raw.get("path"),
            "bytes": raw.get("bytes"),
            "width": raw.get("width"),
            "height": raw.get("height"),
        }
        meta = {k: v for k, v in raw.items()
                if k not in {"id", "path", "bytes", "width", "height"}}
        return Item(id=item_id, content_type=self.content_type,
                    payload=payload, metadata=meta)

    def preprocess(self, item: Item) -> Item:
        # Real impl: decode, EXIF-orient, resize, normalize.
        item.metadata.setdefault("normalized", True)
        return item

    def featurize(self, item: Item) -> Item:
        # Real impl: run a vision backbone -> embedding vector.
        w = item.payload.get("width") or 0
        h = item.payload.get("height") or 0
        item.features = {"aspect_ratio": (w / h) if h else None}
        return item
