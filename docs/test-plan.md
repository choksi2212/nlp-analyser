# LexiLab — Test Plan & Quality Assurance Report

**Document ID:** LXL-TST-001  
**Version:** 1.0  
**Test Suite Status:** 54 / 54 Tests Passing (100% Pass Rate)  
**Execution Runtime:** Python 3.11, pytest, ruff  

---

## 1. Testing Strategy

LexiLab implements a multi-tier testing strategy ensuring algorithm correctness, contract compliance, error handling, and visual rendering safety:

```text
               ┌────────────────────────┐
               │ Acceptance Suite (T01) │  (20 tests)
               └───────────┬────────────┘
                           │
               ┌───────────▼────────────┐
               │ Component & UI Safety  │  (4 tests)
               └───────────┬────────────┘
                           │
               ┌───────────▼────────────┐
               │ Sample Datasets Tests  │  (5 tests)
               └───────────┬────────────┘
                           │
               ┌───────────▼────────────┐
               │ Core Engine Unit Tests │  (25 tests)
               └────────────────────────┘
```

---

## 2. Acceptance Matrix Verification Report (T-01 to T-20)

| ID | Feature | Test Condition / Input | Verified Invariant | Status |
|---|---|---|---|---|
| **T-01** | Validation | Empty string | Raises `InputValidationError` (`EMPTY_TEXT`) | PASSED |
| **T-02** | Validation | 10,001 characters | Rejected with `TEXT_TOO_LONG` | PASSED |
| **T-03** | Preprocessing | `"The students are running."` | Returns tokens, filtered words, stems, lemmas, POS | PASSED |
| **T-04** | Preprocessing | Punctuation-only (`"!!!"`) | No crash; empty alphabetic word list; tokens preserved | PASSED |
| **T-05** | NER | `"Apple opened an office in Mumbai."` | Valid entity spans (`Apple`, `Mumbai`); exact substring match | PASSED |
| **T-06** | NER | `"the and of"` | Returns empty entity list cleanly without error | PASSED |
| **T-07** | Sentiment | `"I absolutely love this!"` | Compound $\ge 0.05$; Label = `"Positive"` | PASSED |
| **T-08** | Sentiment | `"This is terrible."` | Compound $\le -0.05$; Label = `"Negative"` | PASSED |
| **T-09** | Sentiment | Valid text | All sub-scores $\in [0, 1]$; compound $\in [-1, 1]$ | PASSED |
| **T-10** | Summarization| Single sentence input | Friendly error (`INSUFFICIENT_SENTENCES`) | PASSED |
| **T-11** | Summarization| 5-sentence paragraph | Output composed strictly of original sentences | PASSED |
| **T-12** | Summarization| Duplicate sentences | Stable ranking; no crash | PASSED |
| **T-13** | Similarity | Identical text in A and B | Score $\approx 1.0$ ($100\%$) | PASSED |
| **T-14** | Similarity | Completely disjoint vocabularies | Score $\approx 0.0$ ($0\%$) | PASSED |
| **T-15** | Similarity | Inputs reversed ($A \leftrightarrow B$) | Score is exactly symmetric: $\text{sim}(A, B) == \text{sim}(B, A)$ | PASSED |
| **T-16** | Similarity | Stop-word-only text | Caught gracefully (`EMPTY_VOCABULARY`) | PASSED |
| **T-17** | UI Navigation| Switch tabs | Pure functions; no mutating side effects | PASSED |
| **T-18** | Security | `<script>alert(1)</script>` | Inert text; escaped in UI; no JavaScript execution | PASSED |
| **T-19** | Offline | Network disconnected | All 5 modules execute 100% locally | PASSED |
| **T-20** | Privacy | Sample text execution | No user text written to disk, telemetry, or logs | PASSED |

---

## 3. Automated Test Execution Commands

```bash
# Execute entire test suite
python -m pytest tests/

# Execute test suite with verbose output
python -m pytest -v tests/

# Execute linter and code quality checks
python -m ruff check .
python -m ruff format --check .
```
