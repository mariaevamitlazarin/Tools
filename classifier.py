"""
Support Ticket Classifier — basic version
Uses a zero-shot model from HuggingFace transformers.
No training needed: you give it the categories, it picks the best one.

Run:
    pip install transformers torch
    python classifier.py
"""

from transformers import pipeline

# Categories we want to sort tickets into.
CATEGORIES = ["billing", "technical issue", "account access", "feature request", "general question"]

# Load the model once (downloads on first run).
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def classify(ticket_text: str):
    """Return the best category and a confidence score for a ticket."""
    result = classifier(ticket_text, CATEGORIES)
    best_label = result["labels"][0]
    best_score = result["scores"][0]
    return best_label, best_score


if __name__ == "__main__":
    # A few example tickets to try it out.
    tickets = [
        "I was charged twice this month, can I get a refund?",
        "The app crashes every time I open the dashboard.",
        "I forgot my password and can't log in.",
        "It would be great if you added a dark mode.",
    ]

    for ticket in tickets:
        label, score = classify(ticket)
        print(f"Ticket: {ticket}")
        print(f"  -> {label} ({score:.0%} confident)\n")
