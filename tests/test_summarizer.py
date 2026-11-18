"""Unit and contract tests for the extractive summarization engine."""

import pytest

from core.validation import InputValidationError
from nlp.summarizer import summarize_text


def test_summarize_insufficient_sentences(nlp):
    single_sentence = "This is just one single standalone sentence."
    with pytest.raises(InputValidationError) as exc_info:
        summarize_text(single_sentence, nlp)
    assert exc_info.value.code == "INSUFFICIENT_SENTENCES"


def test_summarize_multi_sentence_passage(nlp):
    passage = (
        "Artificial intelligence is transforming diverse industries globally. "
        "Healthcare systems leverage computer vision to diagnose complex medical conditions early. "
        "Financial institutions apply anomaly detection to intercept fraudulent transactions in real time. "
        "Autonomous transportation relies on advanced sensor fusion and predictive deep learning. "
        "Educational institutions employ natural language models to personalize learning curricula."
    )
    res = summarize_text(passage, nlp, max_sentences=2)

    assert res["module"] == "summarizer"
    data = res["result"]
    assert data["original_sentence_count"] == 5
    assert data["summary_sentence_count"] == 2
    assert data["k_used"] == 2
    assert data["summary_word_count"] < data["original_word_count"]
    assert data["compression_percentage"] > 0

    # The summary text must consist entirely of original sentences
    summary = data["summary"]
    assert len(summary) > 0
    for s_idx in data["selected_indices"]:
        orig_s = data["sentence_ranks"][s_idx]["sentence"]
        assert orig_s in summary


def test_summarize_preserves_sentence_order(nlp):
    passage = (
        "First step is problem analysis and data collection. "
        "Second step is cleaning and preprocessing textual features. "
        "Third step is feature engineering using TF-IDF matrices. "
        "Fourth step is evaluating classification accuracy and precision."
    )
    res = summarize_text(passage, nlp, max_sentences=2)
    selected_indices = res["result"]["selected_indices"]
    # Invariant: selected indices must be sorted ascending to preserve narrative flow
    assert selected_indices == sorted(selected_indices)


def test_summarize_k_clamping(nlp):
    passage = "Sentence one is about nature. Sentence two is about weather."
    # With 2 sentences, requesting 3 should clamp to 1 (since k < total_sentences)
    res = summarize_text(passage, nlp, max_sentences=3)
    assert res["result"]["k_used"] == 1
    assert res["result"]["summary_sentence_count"] == 1


def test_summarize_empty_vocabulary(nlp):
    passage = "The and is. But or if."
    with pytest.raises(InputValidationError) as exc_info:
        summarize_text(passage, nlp)
    assert exc_info.value.code == "EMPTY_VOCABULARY"
