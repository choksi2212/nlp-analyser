"""Pytest fixtures for LexiLab test suite."""

import pytest
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer


@pytest.fixture(scope="session")
def nlp():
    """Session-scoped spaCy en_core_web_sm pipeline."""
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="session")
def vader():
    """Session-scoped NLTK VADER analyzer."""
    return SentimentIntensityAnalyzer()
