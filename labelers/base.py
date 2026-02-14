"""Base class for labelers — the 'machine' half that proposes labels."""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.schema import Item, Prediction
from tasks.base import TaskAdapter


class Labeler(ABC):
    name: str = "base"

    @abstractmethod
    def propose(self, item: Item, task: TaskAdapter) -> Prediction:
        """Propose a label for `item` under the rules of `task`."""
