"""LexiLab — Desktop browser application entry point."""

from pathlib import Path

import streamlit as st

from core.constants import APP_SUBTITLE, APP_TITLE
from core.resources import get_spacy_model, get_vader_analyzer
from ui.pages import (
    render_about_page,
    render_entities_page,
    render_overview_page,
    render_preprocessing_page,
    render_sentiment_page,
    render_similarity_page,
    render_summarization_page,
)


def load_custom_css() -> None:
    """Inject custom design tokens and styling from assets/styles.css."""
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def main() -> None:
    """Configure Streamlit layout, initialize cached models, and route navigation."""
    st.set_page_config(
        page_title=f"{APP_TITLE} — {APP_SUBTITLE}",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_custom_css()

    # Pre-cache or load models with friendly spinner feedback
    nlp = get_spacy_model()
    vader = get_vader_analyzer()

    # Sidebar branding & navigation
    with st.sidebar:
        st.title(f"🧠 {APP_TITLE}")
        st.caption(APP_SUBTITLE)
        st.markdown("---")

        page_selection = st.radio(
            "Explore Modules",
            options=[
                "🏠 Overview",
                "🔍 1. Preprocessing",
                "🏷️ 2. Named Entities",
                "💬 3. Sentiment Analysis",
                "📝 4. Extractive Summarization",
                "⚖️ 5. Document Similarity",
                "ℹ️ About & Architecture",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown(
            """
            **🔒 Local-First Architecture**  
            Zero remote telemetry.  
            All models execute offline.
            """
        )

    # Route navigation
    try:
        if page_selection == "🏠 Overview":
            render_overview_page()
        elif page_selection == "🔍 1. Preprocessing":
            render_preprocessing_page(nlp)
        elif page_selection == "🏷️ 2. Named Entities":
            render_entities_page(nlp)
        elif page_selection == "💬 3. Sentiment Analysis":
            render_sentiment_page(vader)
        elif page_selection == "📝 4. Extractive Summarization":
            render_summarization_page(nlp)
        elif page_selection == "⚖️ 5. Document Similarity":
            render_similarity_page()
        elif page_selection == "ℹ️ About & Architecture":
            render_about_page()
    except Exception as exc:  # noqa: BLE001 (top-level application error boundary)
        st.error(
            "An unexpected internal error occurred during analysis. "
            "Please check that input text contains valid English characters."
        )
        st.caption(f"Error category: {type(exc).__name__}")


if __name__ == "__main__":
    main()
