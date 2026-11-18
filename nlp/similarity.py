"""Document similarity engine using TF-IDF vectorization and cosine similarity."""

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.constants import ALGO_SIMILARITY, make_envelope
from core.validation import InputValidationError, validate_text


def compare_documents(text_a: str, text_b: str) -> dict[str, Any]:
    """Compare two English texts using TF-IDF unigram+bigram representation and cosine distance.

    Args:
        text_a: First English text.
        text_b: Second English text.

    Returns:
        Canonical result envelope with similarity score, percentage, and shared terms breakdown.

    Raises:
        InputValidationError: If either text is invalid or vocabulary is empty.
    """
    raw_a = validate_text(text_a, field_name="Text A")
    raw_b = validate_text(text_b, field_name="Text B")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
    )

    try:
        matrix = vectorizer.fit_transform([raw_a, raw_b])
    except ValueError as exc:
        raise InputValidationError(
            "One or both texts contain no usable English words after stop-word removal. "
            "Please provide texts containing descriptive words.",
            code="EMPTY_VOCABULARY",
        ) from exc

    raw_score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    clamped_score = max(0.0, min(1.0, raw_score))
    similarity_score = round(clamped_score, 4)
    similarity_percentage = round(similarity_score * 100, 1)

    # Extract shared features (non-zero in both row 0 and row 1)
    feature_names = vectorizer.get_feature_names_out()
    row_a = matrix.getrow(0).toarray()[0]
    row_b = matrix.getrow(1).toarray()[0]

    shared_features: list[dict[str, Any]] = []
    for idx, (val_a, val_b) in enumerate(zip(row_a, row_b)):
        if val_a > 0 and val_b > 0:
            shared_features.append(
                {
                    "term": feature_names[idx],
                    "weight_a": round(float(val_a), 4),
                    "weight_b": round(float(val_b), 4),
                    "combined_weight": round(float(val_a + val_b), 4),
                }
            )

    # Sort shared features descending by combined weight and take top 10
    shared_features.sort(key=lambda item: item["combined_weight"], reverse=True)
    top_shared = shared_features[:10]

    warnings: list[str] = [
        (
            "TF-IDF cosine similarity measures lexical overlap of shared terms and n-grams. "
            "It does not assess paraphrase equivalence, deep semantic meaning, or factual consistency."
        )
    ]

    total_chars = len(raw_a) + len(raw_b)
    result = {
        "score": similarity_score,
        "percentage": similarity_percentage,
        "shared_terms": top_shared,
        "total_shared_count": len(shared_features),
        "text_a_words": len(raw_a.split()),
        "text_b_words": len(raw_b.split()),
    }

    return make_envelope(
        module="similarity",
        algorithm=ALGO_SIMILARITY,
        input_characters=total_chars,
        result=result,
        warnings=warnings,
    )
