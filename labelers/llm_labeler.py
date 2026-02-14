"""LLM labeler (zero/few-shot).

This is a stub: `_call_model` is where you put a real API call. It is injected
so the pipeline stays testable offline. The default raises, forcing callers to
either inject a client or use a different labeler in tests.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from core.schema import Item, Label, Prediction
from labelers.base import Labeler
from tasks.base import TaskAdapter

# Given a prompt string, return the model's raw text completion.
ModelFn = Callable[[str], str]


class LLMLabeler(Labeler):
    name = "llm"

    def __init__(self, model_fn: Optional[ModelFn] = None, few_shot: str = ""):
        self.model_fn = model_fn
        self.few_shot = few_shot

    def _build_prompt(self, item: Item, task: TaskAdapter) -> str:
        classes = task.label_space().classes
        content = item.payload if isinstance(item.payload, str) else str(item.payload)
        return (
            "You are a strict classifier. Respond ONLY with JSON of the form "
            '{"label": <one of the classes>, "confidence": <0..1>}.\n'
            f"Classes: {classes}\n"
            f"{self.few_shot}\n"
            f"Content: {content}\n"
        )

    def _call_model(self, prompt: str) -> str:
        if self.model_fn is None:
            raise RuntimeError(
                "LLMLabeler has no model_fn. Inject one, e.g. a function that "
                "calls the Anthropic API and returns the completion text."
            )
        return self.model_fn(prompt)

    def propose(self, item: Item, task: TaskAdapter) -> Prediction:
        prompt = self._build_prompt(item, task)
        raw_text = self._call_model(prompt)
        try:
            parsed = json.loads(raw_text)
            value = parsed["label"]
            confidence = float(parsed.get("confidence", 0.5))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            value = task.label_space().classes[0]
            confidence = 0.0     # unparsable -> force human review
        label = Label(value=value, task_type=task.task_type)
        return Prediction(
            item_id=item.id, label=label, confidence=confidence,
            labeler_name=self.name, raw=raw_text,
        )
