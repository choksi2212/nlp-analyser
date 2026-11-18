"""Unit and contract tests for the sentiment analysis engine."""

import pytest

from nlp.sentiment import analyze_sentiment


def test_sentiment_positive(vader):
    text = "I absolutely love this fantastic and magnificent software!"
    res = analyze_sentiment(text, vader)

    assert res["module"] == "sentiment"
    data = res["result"]
    assert data["label"] == "Positive"
    assert data["compound"] >= 0.05
    assert data["scores"]["pos"] > 0.0


def test_sentiment_negative(vader):
    text = "This product was dreadful, painful, and a complete disaster."
    res = analyze_sentiment(text, vader)

    assert res["module"] == "sentiment"
    data = res["result"]
    assert data["label"] == "Negative"
    assert data["compound"] <= -0.05
    assert data["scores"]["neg"] > 0.0


def test_sentiment_neutral(vader):
    text = "The table is made of pine wood."
    res = analyze_sentiment(text, vader)

    assert res["module"] == "sentiment"
    data = res["result"]
    assert data["label"] == "Neutral"
    assert -0.05 < data["compound"] < 0.05


def test_sentiment_score_invariants(vader):
    text = "Good morning everyone. I hope you have a pleasant day."
    res = analyze_sentiment(text, vader)
    scores = res["result"]["scores"]

    assert -1.0 <= scores["compound"] <= 1.0
    assert 0.0 <= scores["pos"] <= 1.0
    assert 0.0 <= scores["neu"] <= 1.0
    assert 0.0 <= scores["neg"] <= 1.0
    assert sum([scores["pos"], scores["neu"], scores["neg"]]) == pytest.approx(
        1.0, abs=0.01
    )
