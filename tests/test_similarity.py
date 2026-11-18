"""Unit and contract tests for the document similarity engine."""

import pytest

from core.validation import InputValidationError
from nlp.similarity import compare_documents


def test_similarity_identical():
    text = "Natural language processing analyzes written language using computational models."
    res = compare_documents(text, text)

    assert res["module"] == "similarity"
    assert res["result"]["score"] == pytest.approx(1.0, abs=1e-4)
    assert res["result"]["percentage"] == pytest.approx(100.0, abs=0.1)
    assert len(res["result"]["shared_terms"]) > 0


def test_similarity_disjoint():
    text_a = "Photosynthesis creates glucose and oxygen in green plants using sunlight."
    text_b = "Microprocessors compute digital binary arithmetic instructions inside silicon transistors."
    res = compare_documents(text_a, text_b)

    assert res["result"]["score"] == pytest.approx(0.0, abs=1e-4)
    assert res["result"]["percentage"] == pytest.approx(0.0, abs=0.1)
    assert len(res["result"]["shared_terms"]) == 0


def test_similarity_symmetry():
    text_a = "The quick brown fox jumps over a sleeping lazy dog."
    text_b = "A brown fox quickly leaps across the lazy hound."

    res_ab = compare_documents(text_a, text_b)["result"]["score"]
    res_ba = compare_documents(text_b, text_a)["result"]["score"]

    assert res_ab == pytest.approx(res_ba, abs=1e-6)


def test_similarity_empty_vocabulary():
    text_a = "The and or if."
    text_b = "By for with in."
    with pytest.raises(InputValidationError) as exc_info:
        compare_documents(text_a, text_b)
    assert exc_info.value.code == "EMPTY_VOCABULARY"
