"""Preprocessing engine: tokenization, stop-word removal, stemming, lemmatization, and POS tagging."""

from typing import Any

from nltk.stem import PorterStemmer
from spacy.language import Language

from core.constants import ALGO_PREPROCESSING, make_envelope
from core.validation import validate_text

_stemmer = PorterStemmer()


def analyze_preprocessing(text: str, nlp: Language) -> dict[str, Any]:
    """Execute complete linguistic preprocessing pipeline on English text.

    Args:
        text: Input English text.
        nlp: Loaded spaCy English language pipeline.

    Returns:
        Canonical result envelope with tokens, words, filtered words, stems,
        lemmas, POS tags, and statistical breakdown.
    """
    raw_text = validate_text(text, field_name="Text")
    doc = nlp(raw_text)

    # 1. Full token stream (including punctuation)
    all_tokens = [t.text for t in doc]

    # 2. Sentences count
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    sentence_count = len(sentences)

    # 3. Alphabetic tokens (words)
    words = [t for t in doc if t.is_alpha]
    word_texts = [t.text for t in words]

    # 4. Stop-word filtering
    stop_words = [t.text for t in words if t.is_stop]
    filtered_words = [t for t in words if not t.is_stop]
    filtered_texts = [t.text for t in filtered_words]

    # 5. Stemming (NLTK PorterStemmer applied to lowercase filtered words)
    stems = [_stemmer.stem(t.text.lower()) for t in filtered_words]

    # 6. Lemmatization (spaCy lemma lowercase)
    lemmas = [t.lemma_.lower() for t in filtered_words]

    # 7. POS tags (coarse pos_ and fine-grained tag_)
    pos_tags = [{"text": t.text, "pos": t.pos_, "tag": t.tag_} for t in words]

    # 8. Token-level comparison table for educational deep dive
    token_comparison = []
    for t in doc:
        is_alpha = t.is_alpha
        is_stop = t.is_stop
        stem = _stemmer.stem(t.text.lower()) if is_alpha else "-"
        lemma = t.lemma_.lower() if is_alpha else t.lemma_
        token_comparison.append(
            {
                "token": t.text,
                "is_alpha": is_alpha,
                "is_stop": is_stop,
                "stem": stem,
                "lemma": lemma,
                "pos": t.pos_,
                "tag": t.tag_,
            }
        )

    # 9. Warnings and statistical summary
    warnings: list[str] = []
    if len(words) == 0:
        warnings.append(
            "No alphabetic words detected. All tokens are punctuation, symbols, or numbers."
        )
    elif len(filtered_words) == 0:
        warnings.append("All alphabetic words were classified as English stop words.")

    stats = {
        "character_count": len(raw_text),
        "sentence_count": sentence_count,
        "total_tokens": len(all_tokens),
        "alphabetic_tokens": len(words),
        "stop_word_count": len(stop_words),
        "retained_words_count": len(filtered_words),
    }

    result = {
        "tokens": all_tokens,
        "words": word_texts,
        "filtered_words": filtered_texts,
        "stems": stems,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "token_comparison": token_comparison,
        "stats": stats,
    }

    return make_envelope(
        module="preprocessing",
        algorithm=ALGO_PREPROCESSING,
        input_characters=len(raw_text),
        result=result,
        warnings=warnings,
    )
