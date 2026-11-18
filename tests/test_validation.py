"""Unit tests for LexiLab input validation."""

import pytest

from core.validation import (
    InputValidationError,
    validate_sentence_count,
    validate_text,
)


def test_validate_text_valid_string():
    raw = "  Hello, world!  "
    validated = validate_text(raw)
    # Must preserve exact original input without silent trimming
    assert validated == raw


def test_validate_text_empty():
    with pytest.raises(InputValidationError) as exc_info:
        validate_text("")
    assert exc_info.value.code == "EMPTY_TEXT"

    with pytest.raises(InputValidationError) as exc_info:
        validate_text("   \n\t  ")
    assert exc_info.value.code == "EMPTY_TEXT"


def test_validate_text_non_string():
    with pytest.raises(InputValidationError) as exc_info:
        validate_text(None)  # type: ignore
    assert exc_info.value.code == "INVALID_TYPE"

    with pytest.raises(InputValidationError) as exc_info:
        validate_text(12345)  # type: ignore
    assert exc_info.value.code == "INVALID_TYPE"


def test_validate_text_length_boundary():
    # 10,000 characters should pass
    exact_limit_text = "a" * 10_000
    assert validate_text(exact_limit_text) == exact_limit_text

    # 10,001 characters should raise TEXT_TOO_LONG
    over_limit_text = "a" * 10_001
    with pytest.raises(InputValidationError) as exc_info:
        validate_text(over_limit_text)
    assert exc_info.value.code == "TEXT_TOO_LONG"


def test_validate_text_script_tag_safety():
    script_input = "<script>alert('xss')</script>"
    # Must be accepted as data, not rejected or executed
    result = validate_text(script_input)
    assert result == script_input


def test_validate_sentence_count():
    # At least 2 sentences
    validate_sentence_count(["Sentence one.", "Sentence two."])

    # 1 sentence should fail
    with pytest.raises(InputValidationError) as exc_info:
        validate_sentence_count(["Only one sentence."])
    assert exc_info.value.code == "INSUFFICIENT_SENTENCES"

    # Empty list should fail
    with pytest.raises(InputValidationError) as exc_info:
        validate_sentence_count([])
    assert exc_info.value.code == "INSUFFICIENT_SENTENCES"
