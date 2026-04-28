"""
Support Ticket Classifier — spaCy version (rule/keyword based)
Lighter and faster than transformers. Good for understanding the basics
before moving to ML models.

Run:
    pip install spacy
    python -m spacy download en_core_web_sm
    python classifier_spacy.py
"""

import spacy

nlp = spacy.load("en_core_web_sm")

# Each category has a set of trigger keywords.
KEYWORDS = {
    "billing": {"charge", "charged", "refund", "invoice", "payment", "price", "bill"},
    "technical issue": {"crash", "error", "bug", "broken", "freeze", "slow", "loading"},
    "account access": {"password", "login", "log", "locked", "access", "username"},
    "feature request": {"feature", "add", "would", "suggestion", "improve", "wish"},
}


def classify(ticket_text: str):
    """Score each category by counting matched keyword lemmas."""
    doc = nlp(ticket_text.lower())
    lemmas = {token.lemma_ for token in doc}

    scores = {cat: len(lemmas & words) for cat, words in KEYWORDS.items()}
    best = max(scores, key=scores.get)

    # If nothing matched, fall back to general.
    if scores[best] == 0:
        return "general question", 0
    return best, scores[best]


if __name__ == "__main__":
    tickets = [
        "I was charged twice this month, can I get a refund?",
        "The app crashes every time I open the dashboard.",
        "I forgot my password and can't log in.",
        "It would be great if you added a dark mode.",
    ]

    for ticket in tickets:
        label, hits = classify(ticket)
        print(f"Ticket: {ticket}")
        print(f"  -> {label} ({hits} keyword matches)\n")
