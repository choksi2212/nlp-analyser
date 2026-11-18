"""Sentiment analysis engine using NLTK VADER."""

from typing import Any

from nltk.sentiment import SentimentIntensityAnalyzer

from core.constants import (
    ALGO_SENTIMENT,
    SENTIMENT_NEGATIVE_THRESHOLD,
    SENTIMENT_POSITIVE_THRESHOLD,
    make_envelope,
)
from core.validation import validate_text


def analyze_sentiment(
    text: str, analyzer: SentimentIntensityAnalyzer
) -> dict[str, Any]:
    """Estimate sentiment polarity of English text using NLTK VADER.

    Args:
        text: Input English text.
        analyzer: Preloaded SentimentIntensityAnalyzer instance.

    Returns:
        Canonical result envelope with compound score, component ratios, and classification.
    """
    raw_text = validate_text(text, field_name="Text")
    scores = analyzer.polarity_scores(raw_text)

    compound: float = float(scores["compound"])

    if compound >= SENTIMENT_POSITIVE_THRESHOLD:
        label = "Positive"
    elif compound <= SENTIMENT_NEGATIVE_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    warnings = [
        (
            "VADER is a rule- and lexicon-based heuristic. Scores may not reliably reflect "
            "sarcasm, irony, metaphorical expressions, or domain-specific nuances."
        )
    ]

    result = {
        "label": label,
        "compound": compound,
        "scores": {
            "pos": float(scores["pos"]),
            "neu": float(scores["neu"]),
            "neg": float(scores["neg"]),
            "compound": compound,
        },
    }

    return make_envelope(
        module="sentiment",
        algorithm=ALGO_SENTIMENT,
        input_characters=len(raw_text),
        result=result,
        warnings=warnings,
    )
