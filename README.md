# 🍏 Eva's Tools

> 💡 _A lighthouse for developers, scientists & researchers — small tools that make the everyday a little brighter._

💖 **Here's a clean structure for a semi-automatic content labeling pipeline. "Semi-automatic" means a model proposes labels and humans verify/correct the uncertain ones.**.


> [!tip]
> All informations here aren't real.

# Semi-Automatic Content Labeling Pipeline

A model proposes labels with a confidence score; confident predictions are
auto-accepted, uncertain ones are routed to human review. Corrections feed back
into the labeler. One shared core works across **any content type × any task**
through pluggable adapters — no per-combination pipelines.

## Design

```
Raw → Content adapter → Labeler → Router ─┬─ confident ─→ Store (auto)
                                          └─ uncertain ─→ Review queue → human → Store
                                                                          │
                                                                  Feedback → retrain/refine
```

- **`core/`** — content/task-agnostic: schema, pipeline orchestrator, router,
  store, aggregator, metrics, active learning, registry. Written once.
- **`content/`** — adapters per modality: `text`, `image`, `audio`, `mixed`.
- **`tasks/`** — adapters per task: `single_class`, `multi_label`, `spans`
  (text), `bbox` (image), `ranking`.
- **`labelers/`** — the machine half: `rules`, `classifier`, `llm`, `ensemble`.
- **`review/`** — review queue (priority = least confident first) + gold
  injection for annotator QC.
- **`feedback/`** — capture corrections, build retraining data / few-shot.

The orchestrator speaks only the three abstract interfaces (`ContentAdapter`,
`TaskAdapter`, `Labeler`), so swapping modality or task is a config change.

## Extending

- New content type → add a folder under `content/`, implement `ContentAdapter`,
  register it in `core/registry.py`.
- New task → add a file under `tasks/`, implement `TaskAdapter`, register it.
- New labeler → implement `Labeler.propose`.

Nothing in `core/` changes.

## Run

```bash
python demo.py                                   # offline end-to-end demo
python cli.py config/text_singleclass.yaml --input data.jsonl
```

## Real model hooks

Stubs marked in the code are where production wiring goes:
- `content/*/adapter.py` `featurize` → real embeddings / backbones.
- `labelers/llm_labeler.py` `_call_model` → inject an API client.
- `labelers/classifier_labeler.py` → inject `predict_proba`.
- `feedback/retrainer.py` `fit` → your training routine.
- `core/store.py` → swap in-memory store for a DB / object store.

## Tests

```bash
python -m pytest          # if pytest available
```
