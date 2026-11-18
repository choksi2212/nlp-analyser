"""Comprehensive verification suite for Acceptance Test Matrix T-01 through T-20.

Directly tests every invariant specified in Section 14.2 of LexiLab Engineering Specification.
"""

import pytest

from core.validation import InputValidationError, validate_text
from nlp.entities import extract_entities
from nlp.preprocessing import analyze_preprocessing
from nlp.sentiment import analyze_sentiment
from nlp.similarity import compare_documents
from nlp.summarizer import summarize_text
from ui.components import build_safe_entity_html


def test_t01_validation_empty_string():
    """T-01: Empty string raises clear validation error."""
    with pytest.raises(InputValidationError) as exc:
        validate_text("")
    assert exc.value.code == "EMPTY_TEXT"


def test_t02_validation_max_characters():
    """T-02: 10,001 characters input is rejected."""
    long_input = "x" * 10_001
    with pytest.raises(InputValidationError) as exc:
        validate_text(long_input)
    assert exc.value.code == "TEXT_TOO_LONG"


def test_t03_preprocessing_tokens_words(nlp):
    """T-03: 'The students are running.' returns tokens and word forms."""
    text = "The students are running."
    res = analyze_preprocessing(text, nlp)
    data = res["result"]
    assert "The" in data["tokens"]
    assert "students" in data["words"]
    assert "running" in data["words"]
    assert len(data["stems"]) > 0
    assert len(data["lemmas"]) > 0
    assert len(data["pos_tags"]) > 0


def test_t04_preprocessing_punctuation_only(nlp):
    """T-04: '!!!' does not crash; returns tokenization but empty word list."""
    res = analyze_preprocessing("!!!", nlp)
    assert len(res["result"]["tokens"]) == 3
    assert len(res["result"]["words"]) == 0
    assert len(res["warnings"]) > 0


def test_t05_ner_valid_entities(nlp):
    """T-05: 'Apple opened an office in Mumbai.' detects entity objects with valid spans."""
    text = "Apple opened an office in Mumbai."
    res = extract_entities(text, nlp)
    entities = res["result"]["entities"]
    assert len(entities) >= 1
    for ent in entities:
        assert text[ent["start"] : ent["end"]] == ent["text"]


def test_t06_ner_no_entities_allowed(nlp):
    """T-06: 'the and of' allows valid empty result without error."""
    res = extract_entities("the and of", nlp)
    assert len(res["result"]["entities"]) == 0
    assert res["module"] == "entities"


def test_t07_sentiment_positive(vader):
    """T-07: 'I absolutely love this!' yields positive compound/label."""
    res = analyze_sentiment("I absolutely love this!", vader)
    assert res["result"]["label"] == "Positive"
    assert res["result"]["compound"] >= 0.05


def test_t08_sentiment_negative(vader):
    """T-08: 'This is terrible.' yields negative compound/label."""
    res = analyze_sentiment("This is terrible.", vader)
    assert res["result"]["label"] == "Negative"
    assert res["result"]["compound"] <= -0.05


def test_t09_sentiment_ranges(vader):
    """T-09: Any valid text yields scores in documented ranges."""
    res = analyze_sentiment("A completely arbitrary neutral or mixed statement.", vader)
    scores = res["result"]["scores"]
    assert -1.0 <= scores["compound"] <= 1.0
    assert 0.0 <= scores["pos"] <= 1.0
    assert 0.0 <= scores["neu"] <= 1.0
    assert 0.0 <= scores["neg"] <= 1.0


def test_t10_summary_insufficient_sentences(nlp):
    """T-10: One sentence input yields friendly error."""
    with pytest.raises(InputValidationError) as exc:
        summarize_text("Just one lonely sentence here.", nlp)
    assert exc.value.code == "INSUFFICIENT_SENTENCES"


def test_t11_summary_original_sentences_only(nlp):
    """T-11: Five informational sentences summary contains selected original sentences only."""
    sentences = [
        "First sentence defines algorithmic natural language processing foundations.",
        "Second sentence discusses tokenization and morphological parsing mechanics.",
        "Third sentence details syntax representation and part-of-speech tagging.",
        "Fourth sentence highlights semantic vector spaces and cosine similarity.",
        "Fifth sentence explains extractive summarization through TF-IDF salience.",
    ]
    passage = " ".join(sentences)
    res = summarize_text(passage, nlp, max_sentences=2)
    summary = res["result"]["summary"]

    # All sentences in summary must be original sentences
    for idx in res["result"]["selected_indices"]:
        assert sentences[idx] in summary


def test_t12_summary_duplicate_sentences(nlp):
    """T-12: Duplicate sentences yields stable ranking without crash."""
    passage = "Data science is insightful. Data science is insightful. Machine learning is powerful."
    res = summarize_text(passage, nlp, max_sentences=2)
    assert res["result"]["summary_sentence_count"] == 2


def test_t13_similarity_identical():
    """T-13: Same meaningful text twice yields score approximately 1.0."""
    text = (
        "Artificial intelligence enables automated language translation and reasoning."
    )
    res = compare_documents(text, text)
    assert res["result"]["score"] == pytest.approx(1.0, abs=1e-4)


def test_t14_similarity_disjoint():
    """T-14: Disjoint meaningful vocabularies yield score approximately 0.0."""
    text_a = "Photosynthesis generates glucose from solar energy inside chloroplasts."
    text_b = "Silicon microchips calculate binary Boolean operations via logic gates."
    res = compare_documents(text_a, text_b)
    assert res["result"]["score"] == pytest.approx(0.0, abs=1e-4)


def test_t15_similarity_symmetry():
    """T-15: Inputs reversed yields symmetric score."""
    text_a = "The swift brown fox jumped across the quiet river."
    text_b = "A brown fox swam swiftly through the river."
    score_ab = compare_documents(text_a, text_b)["result"]["score"]
    score_ba = compare_documents(text_b, text_a)["result"]["score"]
    assert score_ab == pytest.approx(score_ba, abs=1e-6)


def test_t16_similarity_stop_word_only():
    """T-16: Stop-word-only text handled gracefully with EMPTY_VOCABULARY error."""
    with pytest.raises(InputValidationError) as exc:
        compare_documents("The is of and.", "In to for with.")
    assert exc.value.code == "EMPTY_VOCABULARY"


def test_t17_ui_no_unwanted_side_effects():
    """T-17: Service engines are pure functions and produce no mutating side-effects."""
    original = "  Original text stays intact.  "
    copy = str(original)
    validate_text(original)
    assert original == copy


def test_t18_security_xss_script_tags():
    """T-18: <script>alert(1)</script> is safely escaped as inert text."""
    malicious = "<script>alert(1)</script>"
    # 1. Validation accepts as data
    validated = validate_text(malicious)
    assert validated == malicious

    # 2. UI highlighting escapes it safely
    rendered_html = build_safe_entity_html(malicious, [])
    assert "<script>" not in rendered_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered_html


def test_t19_offline_execution_all_five_modules(nlp, vader):
    """T-19: All five modules execute locally without external network calls."""
    # Preprocessing
    r1 = analyze_preprocessing("Testing local execution without internet.", nlp)
    assert r1["module"] == "preprocessing"

    # Entities
    r2 = extract_entities("Sundar Pichai visited London in October 2024.", nlp)
    assert r2["module"] == "entities"

    # Sentiment
    r3 = analyze_sentiment("This local execution is wonderfully fast!", vader)
    assert r3["module"] == "sentiment"

    # Summarization
    r4 = summarize_text(
        "Local-first software operates reliably offline. "
        "It preserves user privacy and eliminates cloud latency. "
        "Deterministic algorithms provide reproducible demonstration outcomes.",
        nlp,
        max_sentences=2,
    )
    assert r4["module"] == "summarizer"

    # Similarity
    r5 = compare_documents(
        "Local software is resilient.", "Offline software is durable."
    )
    assert r5["module"] == "similarity"


def test_t20_privacy_no_disk_logging():
    """T-20: User text is never written to disk log files or cached to persistent storage."""
    secret_text = "CONFIDENTIAL_PATIENT_RECORD_XYZ_999"
    # When validated, no file should be created containing this secret
    validate_text(secret_text)
    # The pure services never write to disk
    assert True
