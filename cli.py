"""CLI: run a pipeline from a YAML config file.

    python cli.py config/text_singleclass.yaml --input data.jsonl

This wires config -> registry -> pipeline. The labeler here is a placeholder
rules labeler; replace `build_labeler` with your real labeler selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.pipeline import Pipeline
from core.registry import make_content_adapter, make_task_adapter
from core.router import Router
from ingestion.base import IterableSource
from labelers.rules_labeler import RulesLabeler


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("PyYAML not installed. `pip install pyyaml`") from e
    return yaml.safe_load(Path(path).read_text())


def build_labeler(cfg: dict):
    kind = cfg.get("labeler", "rules")
    if kind == "rules":
        # Placeholder: no rules -> everything abstains -> goes to human.
        return RulesLabeler(functions=[])
    raise SystemExit(f"Labeler '{kind}' not wired in cli.py yet.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--input", help="JSONL file of raw records")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    records = []
    if args.input:
        for line in Path(args.input).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))

    pipe = Pipeline(
        source=IterableSource(records),
        content=make_content_adapter(cfg["content_type"]),
        task=make_task_adapter(cfg["task_type"], cfg["classes"]),
        labeler=build_labeler(cfg),
        router=Router(threshold=cfg.get("router", {}).get("threshold", 0.85)),
    )
    pipe.run_auto()

    print(f"Routed {pipe.stats.total} items: "
          f"{pipe.stats.auto} auto, {pipe.stats.human} to human.")
    out = cfg.get("output", {}).get("path", "outputs/dataset.jsonl")
    pipe.store.export_jsonl(out)
    print(f"Auto-accepted labels exported to {out} "
          f"({len(pipe.review_queue)} awaiting human review).")


if __name__ == "__main__":
    main()
