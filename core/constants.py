"""Centralized constants, limits, and configuration defaults for LexiLab."""

from typing import Final

# Application metadata
APP_TITLE: Final[str] = "LexiLab"
APP_SUBTITLE: Final[str] = "Local-First English NLP Analysis Playground"
DOCUMENT_ID: Final[str] = "LXL-ENG-001"
VERSION: Final[str] = "1.0.0"

# Input constraints
MAX_INPUT_CHARACTERS: Final[int] = 10_000
MIN_SUMMARY_SENTENCES: Final[int] = 2
RECOMMENDED_SUMMARY_WORDS: Final[int] = 40
DEFAULT_SUMMARY_SENTENCES: Final[int] = 3
MAX_SUMMARY_SENTENCES: Final[int] = 5

# Sentiment classification thresholds
SENTIMENT_POSITIVE_THRESHOLD: Final[float] = 0.05
SENTIMENT_NEGATIVE_THRESHOLD: Final[float] = -0.05

# Supported entity labels and friendly descriptions
ENTITY_LABEL_DESCRIPTIONS: Final[dict[str, str]] = {
    "PERSON": "People, including fictional characters",
    "NORP": "Nationalities, religious or political groups",
    "FAC": "Buildings, airports, highways, bridges, etc.",
    "ORG": "Companies, agencies, institutions, etc.",
    "GPE": "Countries, cities, states (geopolitical entities)",
    "LOC": "Non-GPE locations, mountain ranges, water bodies",
    "PRODUCT": "Objects, vehicles, foods, etc. (not services)",
    "EVENT": "Named hurricanes, battles, wars, sports events",
    "WORK_OF_ART": "Titles of books, songs, paintings, etc.",
    "LAW": "Named laws or legislation",
    "LANGUAGE": "Any named language",
    "DATE": "Absolute or relative dates or periods",
    "TIME": "Times smaller than a day",
    "PERCENT": "Percentage (including '%')",
    "MONEY": "Monetary values, including unit",
    "QUANTITY": "Measurements (e.g. weight, distance)",
    "ORDINAL": "First, second, third, etc.",
    "CARDINAL": "Numerals that do not fall under another type",
}

# Standardized algorithm labels
ALGO_PREPROCESSING: Final[str] = "spaCy en_core_web_sm + NLTK PorterStemmer"
ALGO_NER: Final[str] = "spaCy en_core_web_sm (Transition-based Parser & NER)"
ALGO_SENTIMENT: Final[str] = (
    "NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner)"
)
ALGO_SUMMARIZER: Final[str] = (
    "Extractive TF-IDF Sentence Salience with Sqrt-Feature Normalization"
)
ALGO_SIMILARITY: Final[str] = (
    "scikit-learn TfidfVectorizer (Uni+Bi-grams) & Cosine Similarity"
)


def make_envelope(
    module: str,
    algorithm: str,
    input_characters: int,
    result: dict,
    warnings: list[str] | None = None,
) -> dict:
    """Construct the canonical standardized result envelope for LexiLab modules."""
    return {
        "module": module,
        "input_characters": input_characters,
        "result": result,
        "warnings": warnings or [],
        "algorithm": algorithm,
    }
