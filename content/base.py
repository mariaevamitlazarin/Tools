"""Base class for content-type adapters (text, image, audio, mixed).

A ContentAdapter knows how to turn a raw source record into a canonical Item,
clean it, and derive features. It knows nothing about labeling tasks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.schema import ContentType, Item


class ContentAdapter(ABC):
    content_type: ContentType

    @abstractmethod
    def load(self, raw: Any) -> Item:
        """Wrap a raw source record into a canonical Item."""

    @abstractmethod
    def preprocess(self, item: Item) -> Item:
        """Clean / normalize / segment. Returns a (possibly new) Item."""

    @abstractmethod
    def featurize(self, item: Item) -> Item:
        """Attach features (embeddings, etc.) to item.features and return it."""

    def prepare(self, raw: Any) -> Item:
        """Convenience: full load -> preprocess -> featurize chain."""
        return self.featurize(self.preprocess(self.load(raw)))
