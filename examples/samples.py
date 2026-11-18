"""Curated sample texts for LexiLab modules to enable one-click demonstration."""

from typing import Final

SAMPLE_PREPROCESSING: Final[str] = (
    "The five quick brown foxes are running better than the lazy dogs in London. "
    "Shouldn't we observe how computational systems analyze words, stems, and linguistic tags?"
)

SAMPLE_NER: Final[str] = (
    "On October 24, 2024, Sundar Pichai announced that Google opened a new artificial "
    "intelligence research lab in London, investing over $3 billion across the United Kingdom "
    "and European Union."
)

SAMPLE_SENTIMENT_POSITIVE: Final[str] = (
    "The application is remarkably smooth, intuitive, delightfully fast, and exceptionally well-engineered!"
)

SAMPLE_SENTIMENT_NEGATIVE: Final[str] = (
    "The software was frustrating, slow, completely broken, full of dreadful bugs, and totally unusable."
)

SAMPLE_SENTIMENT_SARCASTIC: Final[str] = (
    "Oh wonderful, another two-hour emergency status meeting scheduled for 5 PM on a Friday "
    "that definitely could have been a three-sentence email."
)

SAMPLE_SUMMARIZATION: Final[str] = (
    "Natural language processing is an interdisciplinary subfield of computer science and artificial intelligence. "
    "It focuses on the interactions between human languages and computational algorithms. "
    "Classical approaches rely on rule-based linguistic grammars, morphological analysis, and statistical token frequencies. "
    "Modern machine learning techniques employ dense vector representations, neural networks, and attention mechanisms. "
    "In educational and production environments, transparent pipelines allow developers to diagnose intermediate transformations at every stage. "
    "Through systematic processing, systems parse syntax, tag entities, and extract structured knowledge from unstructured textual documents."
)

SAMPLE_SIMILARITY_PAIRS: Final[dict[str, tuple[str, str]]] = {
    "Identical (High Overlap)": (
        "Natural language processing analyzes written language using computational algorithms and statistical token modeling.",
        "Natural language processing analyzes written language using computational algorithms and statistical token modeling.",
    ),
    "Thematically Related (Partial Overlap)": (
        "Machine learning algorithms learn statistical patterns and extract features from large document collections.",
        "Modern learning systems detect statistical patterns and linguistic representations from large textual document datasets.",
    ),
    "Completely Disjoint (Zero Overlap)": (
        "Photosynthesis allows green plants and trees to convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        "Modern microprocessors execute billions of digital instructions every second using semiconductor silicon transistors.",
    ),
}
