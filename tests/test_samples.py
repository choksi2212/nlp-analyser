"""Tests to ensure all curated sample datasets execute successfully."""

from examples.samples import (
    SAMPLE_NER,
    SAMPLE_PREPROCESSING,
    SAMPLE_SENTIMENT_NEGATIVE,
    SAMPLE_SENTIMENT_POSITIVE,
    SAMPLE_SENTIMENT_SARCASTIC,
    SAMPLE_SIMILARITY_PAIRS,
    SAMPLE_SUMMARIZATION,
)
from nlp.entities import extract_entities
from nlp.preprocessing import analyze_preprocessing
from nlp.sentiment import analyze_sentiment
from nlp.similarity import compare_documents
from nlp.summarizer import summarize_text


def test_sample_preprocessing(nlp):
    res = analyze_preprocessing(SAMPLE_PREPROCESSING, nlp)
    assert res["result"]["stats"]["total_tokens"] > 0
    assert len(res["result"]["words"]) > 0


def test_sample_ner(nlp):
    res = extract_entities(SAMPLE_NER, nlp)
    assert res["result"]["total_entities"] > 0


def test_sample_sentiment(vader):
    res_pos = analyze_sentiment(SAMPLE_SENTIMENT_POSITIVE, vader)
    assert res_pos["result"]["label"] == "Positive"

    res_neg = analyze_sentiment(SAMPLE_SENTIMENT_NEGATIVE, vader)
    assert res_neg["result"]["label"] == "Negative"

    res_sarc = analyze_sentiment(SAMPLE_SENTIMENT_SARCASTIC, vader)
    assert "compound" in res_sarc["result"]


def test_sample_summarization(nlp):
    res = summarize_text(SAMPLE_SUMMARIZATION, nlp, max_sentences=3)
    assert res["result"]["summary_sentence_count"] == 3
    assert len(res["result"]["summary"]) > 0


def test_sample_similarity_pairs():
    for name, (text_a, text_b) in SAMPLE_SIMILARITY_PAIRS.items():
        res = compare_documents(text_a, text_b)
        assert 0.0 <= res["result"]["score"] <= 1.0
        if "Identical" in name:
            assert res["result"]["score"] > 0.95
