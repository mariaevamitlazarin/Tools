"""Text content adapter.

`load` accepts either a dict {"id", "text", ...} or a raw string.
Replace `featurize` with a real embedding model when you need similarity/AL.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from content.base import ContentAdapter
from core.schema import ContentType, Item


def _hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


class TextAdapter(ContentAdapter):
    content_type = ContentType.TEXT

    def load(self, raw: Any) -> Item:
        if isinstance(raw, dict):
            text = raw.get("text", "")
            item_id = str(raw.get("id") or _hash_id(text))
            meta = {k: v for k, v in raw.items() if k not in {"id", "text"}}
        else:
            text = str(raw)
            item_id = _hash_id(text)
            meta = {}
        return Item(id=item_id, content_type=self.content_type,
                    payload=text, metadata=meta)

    def preprocess(self, item: Item) -> Item:
        text = item.payload.strip()
        text = re.sub(r"\s+", " ", text)        # collapse whitespace
        item.payload = text
        item.metadata.setdefault("char_len", len(text))
        return item

    def featurize(self, item: Item) -> Item:
        # Placeholder: bag-of-word-lengths vector. Swap for real embeddings.
        tokens = item.payload.lower().split()
        item.features = {
            "n_tokens": len(tokens),
            "vocab": set(tokens),
        }
        return item
