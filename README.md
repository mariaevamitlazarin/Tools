# 🍎 Support Ticket Classifier

By **EVA**

A super simple support ticket / question classifier by category. Start with the basics, then evolve toward more complex setups.

## What it does

Takes the text of a support ticket and sorts it into a category:

- 💳 billing
- 🛠️ technical issue
- 🔑 account access
- ✨ feature request
- ❓ general question

## Two approaches

| File | Method | Needs training? | Best for |
|------|--------|-----------------|----------|
| `classifier.py` | transformers (zero-shot) | No | Smart out-of-the-box results |
| `classifier_spacy.py` | spaCy (keyword matching) | No | Learning the basics, lightweight |

## Setup

**Transformers version:**

```bash
pip install transformers torch
python classifier.py
```

**spaCy version:**

```bash
pip install spacy
python -m spacy download en_core_web_sm
python classifier_spacy.py
```

## Example output

```
Ticket: I was charged twice this month, can I get a refund?
  -> billing (94% confident)
```

## How to evolve from here 🌱

1. **Start** with the spaCy keyword version to understand the problem.
2. **Move up** to the transformers zero-shot model (this repo) — no training needed.
3. **Next:** collect labeled tickets and fine-tune a small model for your own categories.
4. **Then:** add confidence thresholds and route low-confidence tickets to a human.
5. **Later:** build an API around it and connect it to your real ticketing system.

---

Made with 🍎 by EVA
