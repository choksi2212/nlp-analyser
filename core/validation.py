"""Input validation rules, constraints, and custom error types for LexiLab."""

from core.constants import MAX_INPUT_CHARACTERS, MIN_SUMMARY_SENTENCES


class InputValidationError(ValueError):
    """Raised when user input violates LexiLab validation constraints."""

    def __init__(self, message: str, code: str = "VALIDATION_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_text(
    text: str,
    field_name: str = "Text",
    min_chars: int = 1,
    max_chars: int = MAX_INPUT_CHARACTERS,
) -> str:
    """Validate a single text string against basic bounds and types.

    Invariants:
    - Must be a Python string.
    - Must contain non-whitespace characters.
    - Must not exceed max_chars.
    - Preserves exact input content without mutation.

    Raises:
        InputValidationError: If validation fails.
    """
    if not isinstance(text, str):
        raise InputValidationError(
            f"{field_name} must be a valid text string.", code="INVALID_TYPE"
        )

    stripped = text.strip()
    if not stripped or len(stripped) < min_chars:
        raise InputValidationError(
            f"Enter some English {field_name.lower()} to analyze.", code="EMPTY_TEXT"
        )

    if len(text) > max_chars:
        raise InputValidationError(
            f"{field_name} exceeds the maximum limit of {max_chars:,} characters (current length: {len(text):,}).",
            code="TEXT_TOO_LONG",
        )

    return text


def validate_sentence_count(
    sentences: list[str],
    min_sentences: int = MIN_SUMMARY_SENTENCES,
) -> None:
    """Validate that a list of sentences meets the minimum requirement for summarization.

    Raises:
        InputValidationError: If sentence count is below threshold.
    """
    if len(sentences) < min_sentences:
        raise InputValidationError(
            f"Add at least {min_sentences} sentences to summarize (found {len(sentences)}).",
            code="INSUFFICIENT_SENTENCES",
        )
