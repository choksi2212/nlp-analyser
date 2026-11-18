# LexiLab — Local-First English NLP Analysis Playground

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5.svg)](https://spacy.io/)
[![Tests](https://img.shields.io/badge/tests-54%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **LexiLab** is a lightweight, transparent, local-first educational playground that accepts English text and demonstrates five natural language processing workflows: preprocessing, named entity recognition, sentiment analysis, extractive summarization, and document similarity. Built with classical and statistical algorithms, it runs entirely offline on your local machine with zero external cloud APIs or databases, providing clear intermediate outputs designed for academic presentation and viva explanations.

---

## 🚀 Key Features & Algorithmic Engines

1. **Linguistic Preprocessing:**
   - Word and sentence tokenization via spaCy.
   - Stop-word identification and filtering.
   - Side-by-side comparison of **Porter Stemming** (NLTK suffix chopping) versus **Lemmatization** (spaCy morphological analysis).
   - Coarse and fine-grained Part-of-Speech (POS) tagging with interactive data tables.
2. **Named Entity Recognition (NER):**
   - Detects standard named categories (`PERSON`, `ORG`, `GPE`, `LOC`, `DATE`, `MONEY`, etc.) using spaCy's transition-based neural pipeline.
   - Safely renders styled, color-coded inline highlighted passages with an interactive category legend.
   - Emits exact character offsets (`start`, `end`) without fabricating pseudo-confidence scores.
3. **Sentiment Analysis:**
   - Rule-based valence analysis powered by **NLTK VADER**.
   - Computes normalized compound polarity scores ($-1.0$ to $+1.0$) alongside positive, neutral, and negative ratio breakdowns.
   - Interactive horizontal gauge with neutral midpoint and educational caveats on sarcasm.
4. **Extractive Text Summarization:**
   - Deterministic sentence ranking via TF-IDF weights normalized by the square root of non-zero terms:
     $$\text{Score}_i = \frac{\sum \text{TFIDF}_i}{\sqrt{\max(1, \text{nnz}_i)}}$$
   - Selects top $k$ sentences ($1 \le k \le 5$) and restores their chronological reading order.
   - Displays compression percentage and a *"Why these sentences?"* ranking table.
5. **Document Similarity:**
   - Joint unigram and bigram `TfidfVectorizer` with cosine similarity distance computation.
   - Displays percentage lexical overlap and extracts top 10 shared n-grams ranked by combined weight.

---

## 🛠️ System Prerequisites

- **Operating System:** Windows 10/11, macOS, or Linux
- **Python Version:** Python 3.11.x (tested on Python 3.11.0)
- **Git**

---

## 📦 Installation & Setup

### Windows (PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/choksi2212/nlp-analyser.git
cd nlp-analyser

# 2. Create isolated virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Download linguistic models and lexicons
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"
```

### macOS / Linux (Bash)

```bash
# 1. Clone repository
git clone https://github.com/choksi2212/nlp-analyser.git
cd nlp-analyser

# 2. Create isolated virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Download linguistic models and lexicons
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## 🖥️ Running the Application

Launch the Streamlit interactive playground locally:

```bash
streamlit run app.py
```

The application will open automatically in your default desktop browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests & Code Quality

LexiLab includes an automated test suite verifying all 20 acceptance criteria from the engineering specification:

```bash
# Run all 54 tests
python -m pytest tests/

# Run tests with verbose output
python -m pytest -v tests/

# Verify code formatting and linting
python -m ruff check .
python -m ruff format --check .
```

---

## 🔒 Privacy Model & Security

- **100% Local-First:** Once models are installed, no network connection is used. Your text is never transmitted over the internet.
- **Zero Disk Logging:** User input is never logged, stored in temporary files, or retained in persistent databases.
- **XSS & Injection Protection:** User text is treated strictly as data. All dynamic text spans rendered in HTML are sanitized with `html.escape()` prior to insertion.
- **Resource Constraints:** Input text is strictly bounded to 10,000 characters to prevent runaway CPU loops.

---

## 📁 Repository Directory Structure

```text
nlp-analyser/
├── .streamlit/
│   └── config.toml             # Custom light theme and server configuration
├── assets/
│   └── styles.css              # Design system tokens and styling rules
├── core/
│   ├── __init__.py
│   ├── constants.py            # Global constants, limits, thresholds & envelope
│   ├── resources.py            # Cached model and analyzer resource loaders
│   └── validation.py           # Input bounds checking and custom exceptions
├── docs/
│   ├── architecture.md         # Detailed architectural documentation
│   ├── test-plan.md            # Test matrix verification report
│   └── viva-notes.md           # 5-minute presentation script & 12 viva Q&As
├── examples/
│   ├── __init__.py
│   └── samples.py              # Curated demonstration datasets for all 5 modules
├── nlp/
│   ├── __init__.py
│   ├── entities.py             # Pure NER extraction engine
│   ├── preprocessing.py        # Pure linguistic preprocessing engine
│   ├── sentiment.py            # Pure VADER sentiment engine
│   ├── similarity.py           # Pure TF-IDF cosine similarity engine
│   └── summarizer.py           # Pure extractive TF-IDF summarization engine
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared session-scoped pytest fixtures
│   ├── test_acceptance_matrix.py # Complete 20/20 acceptance tests (T-01 - T-20)
│   ├── test_app_smoke.py       # Application entry point smoke test
│   ├── test_components.py      # UI helpers and XSS sanitization tests
│   ├── test_entities.py        # NER unit and contract tests
│   ├── test_preprocessing.py   # Preprocessing unit and contract tests
│   ├── test_samples.py         # Curated sample verification tests
│   ├── test_sentiment.py       # Sentiment unit and contract tests
│   ├── test_similarity.py      # Document similarity unit and contract tests
│   ├── test_summarizer.py      # Summarization unit and contract tests
│   └── test_validation.py      # Input boundary and validation unit tests
├── ui/
│   ├── __init__.py
│   ├── components.py           # Reusable Streamlit visual components
│   └── pages.py                # View controllers for all 7 feature sections
├── app.py                      # Main Streamlit desktop application runner
├── LexiLab_Engineering_Specification.md # Full engineering specification (LXL-ENG-001)
├── requirements.txt            # Package constraints
├── requirements-lock.txt       # Frozen tested environment lockfile
└── README.md                   # Project documentation
```

---

## 🎓 Syllabus Traceability

LexiLab directly implements topics aligned with university Natural Language Processing curricula:

| Unit | Syllabus Topic | LexiLab Implementation |
|---|---|---|
| **Unit 1** | NLP Pipeline, Components & Preprocessing | Centralized validation, modular services, UI separation |
| **Unit 2** | POS Tagging, Morphology & NER | spaCy transition-based tagger, fine tags, entity spans |
| **Unit 3** | Tokenization, Stop Words, Stemming, Lemmas | Comparative tokenization, Porter stemmer vs spaCy lemmatizer |
| **Unit 4** | Classification, Summarization & Extraction | NLTK VADER sentiment classification, extractive TF-IDF ranking |
| **Unit 5** | Document Distance & Similarity Metrics | Joint TF-IDF unigram+bigram cosine similarity |
| **Unit 6** | Machine Translation | *Explicitly out of scope (Non-goal)* |

---

## 📜 Attribution & License

- Built with [Streamlit](https://streamlit.io/), [spaCy](https://spacy.io/), [NLTK](https://www.nltk.org/), and [scikit-learn](https://scikit-learn.org/).
- Models: spaCy `en_core_web_sm` (MIT License), NLTK VADER Lexicon (MIT License).
- Released for educational and research demonstration purposes under the MIT License.
