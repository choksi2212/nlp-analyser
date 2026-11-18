"""Feature pages and UI controllers for LexiLab."""

import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from spacy.language import Language

from core.constants import (
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_SUMMARY_SENTENCES,
    DOCUMENT_ID,
    MAX_INPUT_CHARACTERS,
    MAX_SUMMARY_SENTENCES,
    VERSION,
)
from core.validation import InputValidationError
from examples.samples import (
    SAMPLE_NER,
    SAMPLE_PREPROCESSING,
    SAMPLE_SENTIMENT_NEGATIVE,
    SAMPLE_SENTIMENT_POSITIVE,
    SAMPLE_SENTIMENT_SARCASTIC,
    SAMPLE_SIMILARITY_PAIRS,
    SAMPLE_SUMMARIZATION,
)
from nlp.entities import extract_entities
from nlp.preprocessing import analyze_preprocessing
from nlp.sentiment import analyze_sentiment
from nlp.similarity import compare_documents
from nlp.summarizer import summarize_text
from ui.components import (
    render_entity_highlights,
    render_explanation_box,
    render_header,
    render_metric_row,
    render_sentiment_gauge,
    render_token_chips,
    render_warnings,
)


def render_overview_page() -> None:
    """Render the welcome and architecture overview page."""
    render_header(APP_TITLE, APP_SUBTITLE, badge="Overview")

    st.markdown(
        """
        Welcome to **LexiLab**, a lightweight, transparent, local-first playground for inspecting 
        natural language processing algorithms. Unlike black-box generative AI models, LexiLab 
        is designed for **explainability**, allowing you to observe every step of classical 
        and statistical text processing pipelines.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### 🛠️ Interactive NLP Modules
            1. **Preprocessing:** Tokenization, stop-word removal, Porter stemming, spaCy lemmatization, and POS tagging.
            2. **Named Entity Recognition:** Identify and highlight real-world entities (people, organizations, locations, dates).
            3. **Sentiment Analysis:** Transparent lexical sentiment scoring using the NLTK VADER rule-based engine.
            4. **Extractive Summarization:** Salience-driven sentence ranking via TF-IDF weight summation with feature normalization.
            5. **Document Similarity:** Bag-of-words unigram and bigram cosine similarity comparison between two texts.
            """
        )

    with col2:
        st.markdown(
            """
            ### 🔒 Local-First & Viva-Ready Guarantees
            - **100% Offline Analysis:** All computation runs locally on your machine. Zero text is sent to cloud APIs.
            - **Zero Databases or Logins:** No account creation, database, or API keys required.
            - **Deterministic & Interpretable:** Every transformation produces visible intermediate outputs suitable for academic vivas.
            - **Input Safety:** Strict character bounds and safe HTML escaping prevent runaway processes and script injections.
            """
        )

    st.markdown("---")
    st.info("👈 Select any module from the sidebar navigation to begin analyzing text.")


def render_preprocessing_page(nlp: Language) -> None:
    """Render the linguistic preprocessing page."""
    render_header(
        "1. Linguistic Preprocessing",
        "Tokenization, stop words, stemming, lemmatization, and POS tagging",
        badge="Pipeline",
    )

    if "preproc_text" not in st.session_state:
        st.session_state["preproc_text"] = SAMPLE_PREPROCESSING

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Load Sample Text", key="btn_sample_preproc"):
            st.session_state["preproc_text"] = SAMPLE_PREPROCESSING
    with col_btn2:
        if st.button("Clear Input", key="btn_clear_preproc"):
            st.session_state["preproc_text"] = ""

    with st.form("form_preprocessing"):
        user_input = st.text_area(
            "Enter English text to preprocess:",
            value=st.session_state["preproc_text"],
            height=120,
            max_chars=MAX_INPUT_CHARACTERS,
            help=f"Maximum {MAX_INPUT_CHARACTERS:,} characters. Enter English sentences.",
        )
        st.caption(f"Character count: {len(user_input)} / {MAX_INPUT_CHARACTERS:,}")
        submitted = st.form_submit_button("Analyze Preprocessing", type="primary")

    if submitted:
        st.session_state["preproc_text"] = user_input
        try:
            with st.spinner("Processing linguistic tokens..."):
                envelope = analyze_preprocessing(user_input, nlp)

            data = envelope["result"]
            stats = data["stats"]

            st.success("Preprocessing analysis complete!")
            render_warnings(envelope["warnings"])

            # Summary Metrics Row
            render_metric_row(
                [
                    {"label": "Characters", "value": stats["character_count"]},
                    {"label": "Sentences", "value": stats["sentence_count"]},
                    {"label": "Total Tokens", "value": stats["total_tokens"]},
                    {"label": "Alphabetic Words", "value": stats["alphabetic_tokens"]},
                    {"label": "Stop Words", "value": stats["stop_word_count"]},
                    {"label": "Retained Words", "value": stats["retained_words_count"]},
                ]
            )

            # Tabs for viewing transformations
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                [
                    "Tokens",
                    "Stop Words",
                    "Stem vs Lemma",
                    "POS Tags",
                    "Detailed Breakdown",
                ]
            )

            with tab1:
                st.markdown("#### Full Token Stream (Including Punctuation)")
                render_token_chips(data["tokens"])

            with tab2:
                st.markdown("#### Filtered Content Words (Stop Words Removed)")
                render_token_chips(data["filtered_words"])

            with tab3:
                st.markdown("#### Stemming (NLTK Porter) vs Lemmatization (spaCy)")
                st.caption(
                    "Notice how stemming applies suffix-stripping rules while lemmatization finds dictionary base forms."
                )
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Stems (Heuristic Suffix-Chop):**")
                    render_token_chips(data["stems"], chip_type="stem")
                with cols[1]:
                    st.markdown("**Lemmas (Linguistic Base Form):**")
                    render_token_chips(data["lemmas"], chip_type="lemma")

            with tab4:
                st.markdown("#### Part-of-Speech (POS) Tags")
                if data["pos_tags"]:
                    st.dataframe(data["pos_tags"], use_container_width=True, height=250)
                else:
                    st.info("No alphabetic words detected to assign POS tags.")

            with tab5:
                st.markdown("#### Token-by-Token Transformation Table")
                st.dataframe(
                    data["token_comparison"], use_container_width=True, height=300
                )

            render_explanation_box(
                algorithm_name=envelope["algorithm"],
                how_it_works=(
                    "**1. Tokenization:** spaCy splits text into discrete linguistic units using rules and prefix/suffix patterns.<br>"
                    "**2. Stop-word Removal:** High-frequency functional words (e.g., *the, is, and*) are flagged using spaCy's stop list.<br>"
                    "**3. Porter Stemming:** A heuristic rule-based algorithm removes common morphological affixes (e.g., *-ing, -es*).<br>"
                    "**4. Lemmatization:** spaCy uses morphological analysis and dictionary lookups to determine canonical lemmas.<br>"
                    "**5. POS Tagging:** A statistical transition-based model predicts part-of-speech tags based on surrounding syntax."
                ),
                caveats=[
                    "Stemming does not verify if output stems are actual dictionary words (e.g. *running* -> *run*, *university* -> *univers*).",
                    "Stop-word removal can remove essential meaning in tasks like sentiment analysis or question answering (e.g., *not*).",
                    "POS tagging is probabilistic and depends on correct punctuation and capitalization.",
                ],
            )
        except InputValidationError as err:
            st.error(f"Validation Error ({err.code}): {err.message}")


def render_entities_page(nlp: Language) -> None:
    """Render the Named Entity Recognition (NER) page."""
    render_header(
        "2. Named Entity Recognition (NER)",
        "Locate and categorize names, organizations, locations, dates, and currency",
        badge="Extraction",
    )

    if "ner_text" not in st.session_state:
        st.session_state["ner_text"] = SAMPLE_NER

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Load Sample Text", key="btn_sample_ner"):
            st.session_state["ner_text"] = SAMPLE_NER
    with col_btn2:
        if st.button("Clear Input", key="btn_clear_ner"):
            st.session_state["ner_text"] = ""

    with st.form("form_ner"):
        user_input = st.text_area(
            "Enter English text to analyze for named entities:",
            value=st.session_state["ner_text"],
            height=120,
            max_chars=MAX_INPUT_CHARACTERS,
        )
        st.caption(f"Character count: {len(user_input)} / {MAX_INPUT_CHARACTERS:,}")
        submitted = st.form_submit_button("Extract Named Entities", type="primary")

    if submitted:
        st.session_state["ner_text"] = user_input
        try:
            with st.spinner("Extracting named entities..."):
                envelope = extract_entities(user_input, nlp)

            data = envelope["result"]
            entities = data["entities"]

            st.success(
                f"Analysis complete! Detected {data['total_entities']} named entities."
            )
            render_warnings(envelope["warnings"])

            # Highlighted Passage Display
            st.markdown("#### Annotated Entity Passage")
            render_entity_highlights(user_input, entities)

            # Tabular breakdown
            if entities:
                st.markdown("#### Entity Details Table")
                st.dataframe(entities, use_container_width=True)

            render_explanation_box(
                algorithm_name=envelope["algorithm"],
                how_it_works=(
                    "**1. Sentence Parsing:** spaCy parses syntactic dependency relationships.<br>"
                    "**2. Transition-Based NER:** Uses a neural transition-based parser to label contiguous tokens with entity categories.<br>"
                    "**3. Span Offsets:** Start and end character offsets locate exact coordinates in original text."
                ),
                caveats=[
                    "NER is statistical; ambiguous words may be misclassified (e.g. *Apple* as ORG vs PRODUCT).",
                    "spaCy's standard `doc.ents` API does not output calibrated confidence probabilities; confidence scores are not fabricated.",
                    "Rare proper nouns or non-standard formatting may be missed.",
                ],
            )
        except InputValidationError as err:
            st.error(f"Validation Error ({err.code}): {err.message}")


def render_sentiment_page(vader: SentimentIntensityAnalyzer) -> None:
    """Render the Sentiment Analysis page."""
    render_header(
        "3. Sentiment Analysis",
        "Estimate emotional polarity using NLTK VADER rule-based lexicon",
        badge="Classification",
    )

    if "sentiment_text" not in st.session_state:
        st.session_state["sentiment_text"] = SAMPLE_SENTIMENT_POSITIVE

    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        if st.button("Sample Positive", key="btn_senti_pos"):
            st.session_state["sentiment_text"] = SAMPLE_SENTIMENT_POSITIVE
    with col2:
        if st.button("Sample Negative", key="btn_senti_neg"):
            st.session_state["sentiment_text"] = SAMPLE_SENTIMENT_NEGATIVE
    with col3:
        if st.button("Sample Sarcastic", key="btn_senti_sarc"):
            st.session_state["sentiment_text"] = SAMPLE_SENTIMENT_SARCASTIC
    with col4:
        if st.button("Clear Input", key="btn_clear_senti"):
            st.session_state["sentiment_text"] = ""

    with st.form("form_sentiment"):
        user_input = st.text_area(
            "Enter English text to analyze sentiment:",
            value=st.session_state["sentiment_text"],
            height=120,
            max_chars=MAX_INPUT_CHARACTERS,
        )
        st.caption(f"Character count: {len(user_input)} / {MAX_INPUT_CHARACTERS:,}")
        submitted = st.form_submit_button("Analyze Sentiment", type="primary")

    if submitted:
        st.session_state["sentiment_text"] = user_input
        try:
            with st.spinner("Calculating sentiment polarity..."):
                envelope = analyze_sentiment(user_input, vader)

            data = envelope["result"]
            scores = data["scores"]

            st.success("Sentiment analysis complete!")
            render_warnings(envelope["warnings"])

            # Gauge and Label
            render_sentiment_gauge(data["compound"], data["label"])

            # Scores Breakdown Row
            render_metric_row(
                [
                    {"label": "Classification", "value": data["label"]},
                    {
                        "label": "Compound Score",
                        "value": f"{data['compound']:+.4f}",
                        "help": "Normalized score from -1.0 (most negative) to +1.0 (most positive)",
                    },
                    {"label": "Positive Ratio", "value": f"{scores['pos']:.1%}"},
                    {"label": "Neutral Ratio", "value": f"{scores['neu']:.1%}"},
                    {"label": "Negative Ratio", "value": f"{scores['neg']:.1%}"},
                ]
            )

            render_explanation_box(
                algorithm_name=envelope["algorithm"],
                how_it_works=(
                    "**1. Lexicon Lookup:** VADER maps words to validated valence ratings (-4 to +4).<br>"
                    "**2. Heuristic Modifiers:** Punctuation (e.g. *!*), capitalization (e.g. *GREAT*), degree adverbs (e.g. *very*), and contrastive conjunctions (e.g. *but*) modify valence.<br>"
                    "**3. Compound Normalization:** Normalizes raw valence sum into $[-1.0, +1.0]$ via $x / \\sqrt{x^2 + 15}$."
                ),
                caveats=[
                    "VADER is not a deep learning model; it cannot understand deep sarcasm or complex context.",
                    "Component proportions (`pos`, `neu`, `neg`) represent text fractions, not calibrated probabilities.",
                ],
            )
        except InputValidationError as err:
            st.error(f"Validation Error ({err.code}): {err.message}")


def render_summarization_page(nlp: Language) -> None:
    """Render the Extractive Summarization page."""
    render_header(
        "4. Extractive Text Summarization",
        "Deterministic TF-IDF sentence salience ranking and compression",
        badge="Condensation",
    )

    if "summary_text" not in st.session_state:
        st.session_state["summary_text"] = SAMPLE_SUMMARIZATION

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Load Sample Passage", key="btn_sample_summary"):
            st.session_state["summary_text"] = SAMPLE_SUMMARIZATION
    with col2:
        if st.button("Clear Input", key="btn_clear_summary"):
            st.session_state["summary_text"] = ""

    with st.form("form_summarization"):
        user_input = st.text_area(
            "Enter English passage (at least 2 sentences):",
            value=st.session_state["summary_text"],
            height=160,
            max_chars=MAX_INPUT_CHARACTERS,
            help="Extractive summarization selects high-salience sentences from your original text.",
        )
        st.caption(f"Character count: {len(user_input)} / {MAX_INPUT_CHARACTERS:,}")

        max_k = st.slider(
            "Number of summary sentences (k):",
            min_value=1,
            max_value=MAX_SUMMARY_SENTENCES,
            value=DEFAULT_SUMMARY_SENTENCES,
            help="Clamped automatically so summary is strictly shorter than source.",
        )
        submitted = st.form_submit_button("Generate Extractive Summary", type="primary")

    if submitted:
        st.session_state["summary_text"] = user_input
        try:
            with st.spinner("Scoring sentence salience..."):
                envelope = summarize_text(user_input, nlp, max_sentences=max_k)

            data = envelope["result"]
            st.success("Extractive summary generated successfully!")
            render_warnings(envelope["warnings"])

            # Summary Metric Row
            render_metric_row(
                [
                    {
                        "label": "Original Sentences",
                        "value": data["original_sentence_count"],
                    },
                    {
                        "label": "Summary Sentences",
                        "value": data["summary_sentence_count"],
                    },
                    {"label": "Original Words", "value": data["original_word_count"]},
                    {"label": "Summary Words", "value": data["summary_word_count"]},
                    {
                        "label": "Compression",
                        "value": f"{data['compression_percentage']:.1f}%",
                    },
                ]
            )

            st.markdown("#### Generated Extractive Summary")
            st.info(data["summary"])

            st.markdown(
                "#### Sentence Salience Ranking Breakdown ('Why These Sentences?')"
            )
            st.caption(
                "Sentences are scored by TF-IDF weight sum normalized by square root of unique terms, then re-sorted in original narrative order."
            )
            st.dataframe(data["sentence_ranks"], use_container_width=True)

            render_explanation_box(
                algorithm_name=envelope["algorithm"],
                how_it_works=(
                    "**1. Sentence Segmentation:** spaCy identifies sentence boundaries.<br>"
                    "**2. TF-IDF Representation:** Builds a sentence-by-term TF-IDF matrix excluding English stop words.<br>"
                    "**3. Length-Normalized Salience:** Scores each sentence via $\\sum \\text{TFIDF} / \\sqrt{\\max(1, \\text{nnz})}$.<br>"
                    "**4. Chronological Re-ordering:** Selects top $k$ sentences and restores original reading order."
                ),
                caveats=[
                    "Extractive summarization cannot paraphrase or synthesize novel sentences; it only extracts verbatim source sentences.",
                    "May omit crucial pronouns or antecedent context if isolated from neighboring sentences.",
                ],
            )
        except InputValidationError as err:
            st.error(f"Validation Error ({err.code}): {err.message}")


def render_similarity_page() -> None:
    """Render the Document Similarity page."""
    render_header(
        "5. Document Similarity",
        "Compare lexical overlap using TF-IDF n-grams and Cosine Distance",
        badge="Comparison",
    )

    if "sim_text_a" not in st.session_state or "sim_text_b" not in st.session_state:
        pair = SAMPLE_SIMILARITY_PAIRS["Thematically Related (Partial Overlap)"]
        st.session_state["sim_text_a"] = pair[0]
        st.session_state["sim_text_b"] = pair[1]

    pair_choice = st.selectbox(
        "Load preset comparison pair:",
        options=["Select preset...", *list(SAMPLE_SIMILARITY_PAIRS.keys())],
        key="select_sample_sim_pair",
    )
    if pair_choice and pair_choice != "Select preset...":
        sel_pair = SAMPLE_SIMILARITY_PAIRS[pair_choice]
        st.session_state["sim_text_a"] = sel_pair[0]
        st.session_state["sim_text_b"] = sel_pair[1]

    with st.form("form_similarity"):
        col1, col2 = st.columns(2)
        with col1:
            text_a = st.text_area(
                "Document Text A:",
                value=st.session_state["sim_text_a"],
                height=140,
                max_chars=MAX_INPUT_CHARACTERS,
            )
            st.caption(f"Text A chars: {len(text_a)}")
        with col2:
            text_b = st.text_area(
                "Document Text B:",
                value=st.session_state["sim_text_b"],
                height=140,
                max_chars=MAX_INPUT_CHARACTERS,
            )
            st.caption(f"Text B chars: {len(text_b)}")

        submitted = st.form_submit_button("Compute Cosine Similarity", type="primary")

    if submitted:
        st.session_state["sim_text_a"] = text_a
        st.session_state["sim_text_b"] = text_b
        try:
            with st.spinner("Vectorizing texts and computing cosine distance..."):
                envelope = compare_documents(text_a, text_b)

            data = envelope["result"]
            st.success("Similarity calculation complete!")
            render_warnings(envelope["warnings"])

            # Similarity Score Cards
            render_metric_row(
                [
                    {
                        "label": "Cosine Score",
                        "value": f"{data['score']:.4f}",
                        "help": "0.0 = completely disjoint vocabulary, 1.0 = identical TF-IDF vector",
                    },
                    {"label": "Lexical Overlap", "value": f"{data['percentage']:.1f}%"},
                    {"label": "Shared Features", "value": data["total_shared_count"]},
                    {"label": "Text A Words", "value": data["text_a_words"]},
                    {"label": "Text B Words", "value": data["text_b_words"]},
                ]
            )

            # Top Shared Terms Table
            if data["shared_terms"]:
                st.markdown(
                    "#### Top Shared Terms & N-Grams (Ranked by Combined TF-IDF Weight)"
                )
                st.dataframe(data["shared_terms"], use_container_width=True)
            else:
                st.info(
                    "No shared terms detected between Text A and Text B after stop-word removal."
                )

            render_explanation_box(
                algorithm_name=envelope["algorithm"],
                how_it_works=(
                    "**1. Joint Vectorization:** Fits a unigram and bigram `TfidfVectorizer` jointly over both texts.<br>"
                    "**2. Cosine Similarity:** Computes dot product divided by magnitude product: $\\frac{A \\cdot B}{\\|A\\| \\|B\\|}$.<br>"
                    "**3. Shared Terms Extraction:** Identifies intersecting non-zero features ranked by weight."
                ),
                caveats=[
                    "Measures lexical overlap; does not detect synonyms or semantic equivalence (e.g. *feline* vs *cat* gets zero overlap without shared terms).",
                    "Stop words are removed; texts sharing only functional words yield an empty vocabulary warning.",
                ],
            )
        except InputValidationError as err:
            st.error(f"Validation Error ({err.code}): {err.message}")


def render_about_page() -> None:
    """Render the architecture, syllabus mapping, and viva preparation page."""
    render_header(
        "About & Viva Preparation",
        f"{DOCUMENT_ID} v{VERSION} — Architecture and Syllabus Traceability",
        badge="Documentation",
    )

    tab1, tab2, tab3 = st.tabs(
        ["Syllabus Traceability", "Viva Prep & FAQs", "System Architecture"]
    )

    with tab1:
        st.markdown("### 📚 Academic Curriculum Alignment")
        st.markdown(
            """
            | Unit | Syllabus Topic | LexiLab Implementation |
            |---|---|---|
            | **Unit 1** | NLP Pipeline, Components & Preprocessing | Centralized validation, modular services, UI separation |
            | **Unit 2** | POS Tagging, Morphology & NER | spaCy transition-based tagger, fine tags, entity spans |
            | **Unit 3** | Tokenization, Stop Words, Stemming, Lemmas | Comparative tokenization, Porter stemmer vs spaCy lemmatizer |
            | **Unit 4** | Classification, Summarization & Extraction | NLTK VADER sentiment classification, extractive TF-IDF ranking |
            | **Unit 5** | Document Distance & Similarity Metrics | Joint TF-IDF unigram+bigram cosine similarity |
            | **Unit 6** | Machine Translation | *Explicitly out of scope (Non-goal)* |
            """
        )

    with tab2:
        st.markdown("### 🎓 Core Viva Questions & Answers")
        viva_qas = [
            (
                "What is the difference between Stemming and Lemmatization?",
                "Stemming is a heuristic, rule-based approach that cuts off word suffixes without understanding grammar or checking if the result is a valid word (e.g., 'university' -> 'univers'). Lemmatization uses complete morphological analysis and dictionary lookups to find the canonical base form (lemma) based on part-of-speech context.",
            ),
            (
                "Why is VADER not considered a deep learning model?",
                "VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule- and lexicon-based model. It relies on a human-validated sentiment lexicon and grammatical heuristic rules (capitalization, punctuation, degree adverbs, negation) rather than neural network weights trained on backpropagation.",
            ),
            (
                "What is the difference between Extractive and Abstractive summarization?",
                "Extractive summarization selects and reorders exact, verbatim sentences from the source text based on salience metrics. Abstractive summarization generates new paraphrased sentences using generative language models, which requires much heavier computation and can produce factual hallucinations.",
            ),
            (
                "What are the limitations of TF-IDF Cosine Similarity?",
                "TF-IDF cosine similarity measures lexical bag-of-words overlap. It cannot capture synonyms or semantic equivalence without shared vocabulary (e.g., 'The physician treated the infant' and 'The doctor cured the baby' will have 0% similarity).",
            ),
            (
                "Why is LexiLab designed local-first without databases?",
                "To ensure zero privacy leaks, zero recurring API costs, reproducible deterministic execution during demonstrations, and resilience against network failures during live academic evaluations.",
            ),
        ]
        for q, a in viva_qas:
            with st.expander(f"❓ {q}"):
                st.markdown(a)

    with tab3:
        st.markdown("### 🏗️ Technical Architecture")
        st.markdown(
            """
            - **Runtime:** Python 3.11
            - **Framework:** Streamlit (Layout, Form controls, Session state)
            - **Linguistic Engine:** spaCy `en_core_web_sm`
            - **Lexicon Engine:** NLTK VADER (`vader_lexicon`)
            - **Vectorization:** scikit-learn `TfidfVectorizer` + `cosine_similarity`
            - **Offline Integrity:** Zero external HTTP requests at runtime.
            """
        )
