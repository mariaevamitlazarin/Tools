"""Audio content adapter.

`load` accepts a dict {"id", "path"/"samples", "sample_rate", "duration", ...}.
Resampling/segmentation/denoise stubbed; wire in librosa/torchaudio and an audio
embedding model for production.
"""
from __future__ import annotations

from typing import Any

from content.base import ContentAdapter
from core.schema import ContentType, Item


class AudioAdapter(ContentAdapter):
    content_type = ContentType.AUDIO

    def load(self, raw: Any) -> Item:
        if not isinstance(raw, dict):
            raise TypeError("AudioAdapter expects a dict record")
        item_id = str(raw.get("id") or raw.get("path", "audio"))
        payload = {
            "path": raw.get("path"),
            "samples": raw.get("samples"),
            "sample_rate": raw.get("sample_rate"),
            "duration": raw.get("duration"),
        }
        meta = {k: v for k, v in raw.items()
                if k not in {"id", "path", "samples", "sample_rate", "duration"}}
        return Item(id=item_id, content_type=self.content_type,
                    payload=payload, metadata=meta)

    def preprocess(self, item: Item) -> Item:
        # Real impl: resample to target SR, trim silence, segment.
        item.metadata.setdefault("target_sr", 16000)
        return item

    def featurize(self, item: Item) -> Item:
        # Real impl: log-mel spectrogram or audio embedding.
        item.features = {"duration": item.payload.get("duration")}
        return item
