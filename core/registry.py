"""Registries: turn config strings into concrete objects.

Adding a new content type or task means registering it here (and writing its
module). Nothing in core/pipeline.py needs to change.
"""
from __future__ import annotations

from content.audio.adapter import AudioAdapter
from content.base import ContentAdapter
from content.image.adapter import ImageAdapter
from content.mixed.adapter import MixedAdapter
from content.text.adapter import TextAdapter
from core.schema import ContentType, LabelSpace, TaskType
from tasks.base import TaskAdapter
from tasks.bbox import BBoxTask
from tasks.multi_label import MultiLabelTask
from tasks.ranking import RankingTask
from tasks.single_class import SingleClassTask
from tasks.spans import SpansTask

_CONTENT = {
    ContentType.TEXT: TextAdapter,
    ContentType.IMAGE: ImageAdapter,
    ContentType.AUDIO: AudioAdapter,
    ContentType.MIXED: MixedAdapter,
}

_TASK = {
    TaskType.SINGLE_CLASS: SingleClassTask,
    TaskType.MULTI_LABEL: MultiLabelTask,
    TaskType.SPANS: SpansTask,
    TaskType.BBOX: BBoxTask,
    TaskType.RANKING: RankingTask,
}


def make_content_adapter(content_type: str | ContentType) -> ContentAdapter:
    return _CONTENT[ContentType(content_type)]()


def make_task_adapter(
    task_type: str | TaskType, classes: list[str]
) -> TaskAdapter:
    tt = TaskType(task_type)
    space = LabelSpace(task_type=tt, classes=classes)
    return _TASK[tt](space)
