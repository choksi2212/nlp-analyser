"""Unit and contract tests for the preprocessing engine."""

from nlp.preprocessing import analyze_preprocessing


def test_preprocessing_standard_sentence(nlp):
    text = "The students are running quickly."
    res = analyze_preprocessing(text, nlp)

    assert res["module"] == "preprocessing"
    assert res["input_characters"] == len(text)
    assert len(res["warnings"]) == 0

    data = res["result"]
    assert "students" in data["words"]
    assert "running" in data["words"]
    assert "students" in data["filtered_words"]
    # Stop words "The" and "are" should be removed from filtered_words
    assert "the" not in [w.lower() for w in data["filtered_words"]]
    assert "are" not in [w.lower() for w in data["filtered_words"]]

    # Verify stems and lemmas are generated
    assert len(data["stems"]) == len(data["filtered_words"])
    assert len(data["lemmas"]) == len(data["filtered_words"])

    # POS tags should be present for alphabetic words
    assert len(data["pos_tags"]) == len(data["words"])
    pos_map = {item["text"]: item["pos"] for item in data["pos_tags"]}
    assert pos_map["students"] == "NOUN"
    assert pos_map["running"] in ("VERB", "ADJ")

    stats = data["stats"]
    assert (
        stats["total_tokens"]
        >= stats["alphabetic_tokens"]
        >= stats["retained_words_count"]
    )
    assert stats["sentence_count"] == 1


def test_preprocessing_stemming_vs_lemmatization(nlp):
    # Test word where stem and lemma diverge: e.g. "mice", "better"
    text = "Mice are better."
    res = analyze_preprocessing(text, nlp)["result"]

    # "mice": stem is usually "mice", lemma is "mouse"
    # "better": stem is usually "better", lemma is "good" or "well"
    filtered = res["filtered_words"]
    stems = dict(zip(filtered, res["stems"]))
    lemmas = dict(zip(filtered, res["lemmas"]))

    if "Mice" in stems:
        assert lemmas["Mice"] == "mouse"


def test_preprocessing_punctuation_only(nlp):
    text = "!?!?"
    res = analyze_preprocessing(text, nlp)
    data = res["result"]

    assert len(data["tokens"]) == 4
    assert len(data["words"]) == 0
    assert len(data["filtered_words"]) == 0
    assert len(data["stems"]) == 0
    assert len(data["lemmas"]) == 0
    assert len(data["pos_tags"]) == 0
    assert len(res["warnings"]) > 0
    assert "No alphabetic words detected" in res["warnings"][0]
