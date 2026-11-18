"""Named Entity Recognition engine using spaCy."""

from typing import Any

from spacy.language import Language

from core.constants import ALGO_NER, make_envelope
from core.validation import validate_text


def extract_entities(text: str, nlp: Language) -> dict[str, Any]:
    """Extract named entities and character spans from English text.

    Args:
        text: Input text to analyze.
        nlp: Loaded spaCy language pipeline.

    Returns:
        Canonical result envelope containing list of entity spans and count by label.
    """
    raw_text = validate_text(text, field_name="Text")
    doc = nlp(raw_text)

    entities: list[dict[str, Any]] = []
    counts_by_label: dict[str, int] = {}

    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
        )
        counts_by_label[ent.label_] = counts_by_label.get(ent.label_, 0) + 1

    warnings: list[str] = []
    if len(entities) == 0:
        warnings.append("No named entities detected in the provided text.")

    result = {
        "entities": entities,
        "counts_by_label": counts_by_label,
        "total_entities": len(entities),
    }

    return make_envelope(
        module="entities",
        algorithm=ALGO_NER,
        input_characters=len(raw_text),
        result=result,
        warnings=warnings,
    )
