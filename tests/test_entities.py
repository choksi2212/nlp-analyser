"""Unit and contract tests for the Named Entity Recognition engine."""

from nlp.entities import extract_entities


def test_extract_entities_standard(nlp):
    text = "Apple opened a new headquarters in London on October 24, 2024."
    res = extract_entities(text, nlp)

    assert res["module"] == "entities"
    assert res["input_characters"] == len(text)
    assert len(res["warnings"]) == 0

    data = res["result"]
    entities = data["entities"]
    assert len(entities) > 0

    # Invariant: extracted character spans must strictly match the text slice
    for ent in entities:
        assert text[ent["start"] : ent["end"]] == ent["text"]
        assert len(ent["label"]) > 0

    # Check label counts match
    counts = data["counts_by_label"]
    assert sum(counts.values()) == len(entities)

    entity_texts = [e["text"] for e in entities]
    assert "Apple" in entity_texts
    assert "London" in entity_texts


def test_extract_entities_none_found(nlp):
    text = "the and of on in with"
    res = extract_entities(text, nlp)

    assert res["module"] == "entities"
    data = res["result"]
    assert len(data["entities"]) == 0
    assert len(data["counts_by_label"]) == 0
    assert len(res["warnings"]) > 0
    assert "No named entities detected" in res["warnings"][0]
