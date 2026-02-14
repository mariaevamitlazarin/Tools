"""Ingestion: pull raw records from a source into the pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator


class IngestionSource(ABC):
    @abstractmethod
    def stream(self) -> Iterator[Any]:
        """Yield raw source records one at a time."""


class IterableSource(IngestionSource):
    """Wrap any in-memory iterable. Useful for tests and small jobs."""

    def __init__(self, records: Iterable[Any]):
        self._records = list(records)

    def stream(self) -> Iterator[Any]:
        yield from self._records
