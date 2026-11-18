"""Reusable UI visual components and safe HTML renderers for LexiLab."""

import html
from typing import Any

import streamlit as st

from core.constants import ENTITY_LABEL_DESCRIPTIONS


def render_header(title: str, subtitle: str, badge: str = "LexiLab") -> None:
    """Render a consistent page header with an accent badge and clear subtitle."""
    st.markdown(
        f'<span class="hero-badge">{html.escape(badge)}</span>', unsafe_allow_html=True
    )
    st.title(title)
    st.markdown(f"*{html.escape(subtitle)}*")
    st.markdown("---")


def render_metric_row(metrics: list[dict[str, Any]]) -> None:
    """Render a responsive horizontal row of summary metrics using Streamlit native columns."""
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        label = m.get("label", "")
        value = m.get("value", "")
        help_text = m.get("help", None)
        col.metric(label=label, value=value, help=help_text)


def render_token_chips(tokens: list[str], chip_type: str = "default") -> None:
    """Render small wrapping chips for inspecting individual token lists."""
    if not tokens:
        st.info("No items to display.")
        return

    css_class = "token-chip"
    if chip_type == "lemma":
        css_class = "token-chip token-chip-lemma"
    elif chip_type == "stem":
        css_class = "token-chip token-chip-stem"
    elif chip_type == "stop":
        css_class = "token-chip token-chip-stop"

    chips_html = "".join(
        f'<span class="{css_class}">{html.escape(t)}</span>' for t in tokens
    )
    st.markdown(
        f'<div class="chip-container">{chips_html}</div>', unsafe_allow_html=True
    )


def build_safe_entity_html(text: str, entities: list[dict[str, Any]]) -> str:
    """Build safely escaped HTML markup with styled <mark> tags for named entities.

    Invariants:
    - All non-entity text segments are escaped with html.escape().
    - All entity texts are escaped with html.escape().
    - Non-overlapping entity spans are sorted by start index.
    - Zero XSS vulnerability.
    """
    if not entities:
        return f'<div class="lexilab-card">{html.escape(text)}</div>'

    # Sort entities by start character ascending
    sorted_ents = sorted(entities, key=lambda e: int(e["start"]))
    html_parts: list[str] = []
    last_idx = 0

    for ent in sorted_ents:
        start = int(ent["start"])
        end = int(ent["end"])
        label = str(ent["label"]).upper()
        label_class = f"entity-{label.lower()}"

        # Append escaped text before the entity span
        if start > last_idx:
            html_parts.append(html.escape(text[last_idx:start]))

        # Append safely escaped entity mark
        ent_text = html.escape(text[start:end])
        safe_label = html.escape(label)
        mark_html = (
            f'<mark class="entity-mark {label_class}">'
            f"{ent_text}"
            f'<span class="entity-label-tag">{safe_label}</span>'
            f"</mark>"
        )
        html_parts.append(mark_html)
        last_idx = end

    # Append any remaining text after the last entity
    if last_idx < len(text):
        html_parts.append(html.escape(text[last_idx:]))

    full_markup = "".join(html_parts)
    return f'<div class="lexilab-card" style="line-height: 2.2; font-size: 1.05rem;">{full_markup}</div>'


def render_entity_highlights(text: str, entities: list[dict[str, Any]]) -> None:
    """Render highlighted entity passage with styled tags and a category legend."""
    st.markdown(build_safe_entity_html(text, entities), unsafe_allow_html=True)

    # Render category legend if entities were detected
    observed_labels = sorted({e["label"] for e in entities})
    if observed_labels:
        with st.expander("Entity Label Legend & Meanings", expanded=False):
            legend_rows = []
            for lbl in observed_labels:
                desc = ENTITY_LABEL_DESCRIPTIONS.get(lbl, "Named entity category")
                legend_rows.append(f"**`{lbl}`**: {desc}")
            st.markdown("\n\n".join(legend_rows))


def render_sentiment_gauge(compound_score: float, label: str) -> None:
    """Render a visual score breakdown and compound polarity gauge."""
    color_map = {
        "Positive": "#16A34A",
        "Negative": "#DC2626",
        "Neutral": "#64748B",
    }
    badge_color = color_map.get(label, "#4F46E5")

    # Map compound [-1.0, 1.0] to gauge percentage [0%, 100%]
    percentage = int(((compound_score + 1.0) / 2.0) * 100)
    percentage = max(0, min(100, percentage))

    gauge_html = f"""
    <div style="margin: 1.5rem 0;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span style="font-weight: 600; font-size: 1.1rem; color: {badge_color};">
          Overall Polarity: {html.escape(label)}
        </span>
        <span style="font-size: 0.95rem; font-family: monospace; color: #64748B;">
          Compound Score: <strong>{compound_score:+.4f}</strong>
        </span>
      </div>
      <div style="background-color: #E2E8F0; border-radius: 9999px; height: 16px; position: relative; overflow: hidden;">
        <div style="background: linear-gradient(90deg, #DC2626 0%, #F59E0B 50%, #16A34A 100%); width: 100%; height: 100%; opacity: 0.4;"></div>
        <div style="position: absolute; top: 0; bottom: 0; left: {percentage}%; width: 4px; background-color: #0F172A; transform: translateX(-50%);"></div>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94A3B8; margin-top: 0.25rem;">
        <span>-1.0 (Most Negative)</span>
        <span>0.0 (Neutral Midpoint)</span>
        <span>+1.0 (Most Positive)</span>
      </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)


def render_warnings(warnings: list[str]) -> None:
    """Render operational notices and algorithm limitation warnings."""
    for warn in warnings:
        st.warning(warn, icon="⚠️")


def render_explanation_box(
    algorithm_name: str,
    how_it_works: str,
    caveats: list[str],
) -> None:
    """Render an expandable educational callout explaining the underlying algorithm and limitations."""
    with st.expander(f"📘 How This Works: {algorithm_name}", expanded=False):
        st.markdown(how_it_works)
        if caveats:
            st.markdown("#### ⚠️ Known Limitations & Educational Notes")
            for cav in caveats:
                st.markdown(f"- {cav}")
