"""Cached resource loaders for expensive linguistic models and lexicons."""

import nltk
import spacy
import spacy.cli
from nltk.sentiment import SentimentIntensityAnalyzer
from spacy.language import Language

from core.validation import InputValidationError

# Attempt to use Streamlit caching if running in a Streamlit context
try:
    import streamlit as st

    cache_resource = st.cache_resource
except ImportError:
    # Fallback to no-op decorator if imported in pure testing / non-Streamlit context
    def cache_resource(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@cache_resource(show_spinner="Loading English NLP pipeline (spaCy en_core_web_sm)...")
def get_spacy_model() -> Language:
    """Load and cache the spaCy English pipeline.

    Retains the standard pipeline components:
    tokenizer, tagger, parser, ner, lemmatizer.
    Attempts automatic download if running in cloud deployment environments.
    """
    try:
        return spacy.load("en_core_web_sm")
    except (OSError, LookupError):
        # Fallback for cloud deployment environments (e.g. Streamlit Cloud)
        try:
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as exc:
            raise InputValidationError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Please run: python -m spacy download en_core_web_sm",
                code="MODEL_NOT_INSTALLED",
            ) from exc


@cache_resource(show_spinner=False)
def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Load and cache the NLTK VADER sentiment analyzer.

    Attempts automatic download of vader_lexicon if missing in cloud deployment.
    """
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception as exc:
            raise InputValidationError(
                "NLTK VADER lexicon is not downloaded. "
                "Please run: python -c \"import nltk; nltk.download('vader_lexicon')\"",
                code="LEXICON_NOT_INSTALLED",
            ) from exc
