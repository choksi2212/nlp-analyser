"""Extractive text summarization engine using TF-IDF sentence salience scoring."""

import math
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.language import Language

from core.constants import (
    ALGO_SUMMARIZER,
    DEFAULT_SUMMARY_SENTENCES,
    make_envelope,
)
from core.validation import InputValidationError, validate_sentence_count, validate_text


def summarize_text(
    text: str,
    nlp: Language,
    max_sentences: int = DEFAULT_SUMMARY_SENTENCES,
) -> dict[str, Any]:
    """Produce an extractive summary by selecting high-salience original sentences.

    Uses TF-IDF term weights normalized by the square root of non-zero terms
    to score sentences, preserving their original narrative order in the summary.

    Args:
        text: Input English text passage.
        nlp: Loaded spaCy pipeline for sentence boundary detection.
        max_sentences: Desired number of sentences in the output (clamped to 1-5).

    Returns:
        Canonical result envelope with summary text, sentence rankings, and compression stats.

    Raises:
        InputValidationError: If input is empty, has <2 sentences, or has empty vocabulary.
    """
    raw_text = validate_text(text, field_name="Passage")
    doc = nlp(raw_text)

    # 1. Split into non-empty sentences
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    validate_sentence_count(sentences, min_sentences=2)

    total_sentences = len(sentences)

    # 2. Clamp k strictly: 1 <= k <= 5, and k < total_sentences so summary is shorter
    k = min(max(1, int(max_sentences)), 5, total_sentences - 1)

    # 3. Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )

    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError as exc:
        raise InputValidationError(
            "Text contains only stop words, punctuation, or numbers. "
            "Please provide sentences containing descriptive English words.",
            code="EMPTY_VOCABULARY",
        ) from exc

    # 4. Compute salience score per sentence: sum(TFIDF) / sqrt(max(1, count_nonzero))
    scores: list[float] = []
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        nnz = int(row.nnz)
        row_sum = float(row.sum())
        norm_score = row_sum / math.sqrt(max(1, nnz)) if nnz > 0 else 0.0
        scores.append(round(norm_score, 4))

    # 5. Rank descending by score; break ties stably by original index ascending
    ranked_indices = sorted(range(total_sentences), key=lambda idx: (-scores[idx], idx))

    # 6. Select top k and sort chronologically
    chosen_indices = sorted(ranked_indices[:k])

    # 7. Join selected original sentences
    summary_sentences = [sentences[i] for i in chosen_indices]
    summary_text = " ".join(summary_sentences)

    # 8. Compute word counts and compression ratio
    original_words = sum(len(s.split()) for s in sentences)
    summary_words = sum(len(s.split()) for s in summary_sentences)
    compression_ratio = 1.0 - (summary_words / max(1, original_words))
    compression_pct = round(max(0.0, compression_ratio) * 100, 1)

    # 9. Sentence breakdown ranking table for inspection
    sentence_ranks = [
        {
            "rank": rank + 1,
            "original_index": idx,
            "score": scores[idx],
            "sentence": sentences[idx],
            "selected": idx in chosen_indices,
        }
        for rank, idx in enumerate(ranked_indices)
    ]

    warnings: list[str] = []
    if total_sentences < 4:
        warnings.append(
            "Passages with fewer than 4 sentences provide limited material for meaningful extractive ranking."
        )
    if int(max_sentences) >= total_sentences:
        warnings.append(
            f"Requested {max_sentences} sentences clamped to {k} to ensure summary is shorter than the source."
        )

    result = {
        "summary": summary_text,
        "selected_indices": chosen_indices,
        "sentence_scores": scores,
        "sentence_ranks": sentence_ranks,
        "k_requested": int(max_sentences),
        "k_used": k,
        "original_sentence_count": total_sentences,
        "summary_sentence_count": len(summary_sentences),
        "original_word_count": original_words,
        "summary_word_count": summary_words,
        "compression_percentage": compression_pct,
    }

    return make_envelope(
        module="summarizer",
        algorithm=ALGO_SUMMARIZER,
        input_characters=len(raw_text),
        result=result,
        warnings=warnings,
    )
