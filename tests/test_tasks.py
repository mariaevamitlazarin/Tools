"""Tests covering routing logic and each task adapter's core behavior."""
from core.router import Router
from core.schema import Label, LabelSpace, Prediction, TaskType
from tasks.bbox import BBoxTask
from tasks.multi_label import MultiLabelTask
from tasks.ranking import RankingTask
from tasks.single_class import SingleClassTask
from tasks.spans import SpansTask


def _pred(label, conf):
    return Prediction(item_id="x", label=label, confidence=conf, labeler_name="t")


def test_router_below_threshold_goes_to_human():
    task = SingleClassTask(LabelSpace(TaskType.SINGLE_CLASS, ["a", "b"]))
    router = Router(threshold=0.8)
    pred = _pred(Label("a", TaskType.SINGLE_CLASS), 0.5)
    assert router.route(pred, task) == "human"


def test_router_invalid_label_goes_to_human_even_if_confident():
    task = SingleClassTask(LabelSpace(TaskType.SINGLE_CLASS, ["a", "b"]))
    router = Router(threshold=0.1)
    pred = _pred(Label("zzz", TaskType.SINGLE_CLASS), 0.99)  # not in classes
    assert router.route(pred, task) == "human"


def test_single_class_consensus_majority():
    task = SingleClassTask(LabelSpace(TaskType.SINGLE_CLASS, ["a", "b"]))
    labels = [Label("a", TaskType.SINGLE_CLASS),
              Label("a", TaskType.SINGLE_CLASS),
              Label("b", TaskType.SINGLE_CLASS)]
    assert task.consensus(labels).value == "a"
    assert abs(task.agreement(labels) - 2 / 3) < 1e-9


def test_multi_label_validate_and_consensus():
    task = MultiLabelTask(LabelSpace(TaskType.MULTI_LABEL, ["x", "y", "z"]))
    assert task.validate(Label(["x", "y"], TaskType.MULTI_LABEL))
    assert not task.validate(Label(["x", "q"], TaskType.MULTI_LABEL))
    labels = [Label(["x", "y"], TaskType.MULTI_LABEL),
              Label(["x"], TaskType.MULTI_LABEL)]
    assert task.consensus(labels).value == ["x"]  # endorsed by >= half


def test_spans_iou_agreement():
    task = SpansTask(LabelSpace(TaskType.SPANS, ["PER"]))
    a = Label([{"start": 0, "end": 4, "tag": "PER"}], TaskType.SPANS)
    b = Label([{"start": 0, "end": 4, "tag": "PER"}], TaskType.SPANS)
    assert task.agreement([a, b]) == 1.0
    assert task.validate(a)


def test_bbox_validate_and_iou():
    task = BBoxTask(LabelSpace(TaskType.BBOX, ["cat"]))
    box = Label([{"x": 0, "y": 0, "w": 10, "h": 10, "tag": "cat"}], TaskType.BBOX)
    assert task.validate(box)
    assert task.agreement([box, box]) == 1.0


def test_ranking_concordance_and_consensus():
    task = RankingTask(LabelSpace(TaskType.RANKING, ["a", "b", "c"]))
    r1 = Label(["a", "b", "c"], TaskType.RANKING)
    r2 = Label(["a", "b", "c"], TaskType.RANKING)
    assert task.agreement([r1, r2]) == 1.0
    r3 = Label(["c", "b", "a"], TaskType.RANKING)
    assert task.consensus([r1, r1, r3]).value[0] == "a"  # Borda favors a
