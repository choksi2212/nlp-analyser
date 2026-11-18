"""Unit tests for UI helper components and HTML sanitization safety."""

from ui.components import build_safe_entity_html


def test_build_safe_entity_html_escapes_xss():
    malicious_text = "<script>alert('pwned')</script> was founded in London."
    entities = [{"text": "London", "label": "GPE", "start": 47, "end": 53}]
    html_output = build_safe_entity_html(malicious_text, entities)

    # Invariant: Raw script tags MUST NOT appear in the rendered HTML output
    assert "<script>" not in html_output
    assert (
        "&lt;script&gt;alert(&#x27;pwned&#x27;)&lt;/script&gt;" in html_output
        or "&lt;script&gt;alert('pwned')&lt;/script&gt;" in html_output
    )
    # Entity mark must be correctly rendered
    assert '<mark class="entity-mark entity-gpe">London' in html_output


def test_build_safe_entity_html_entity_span_escapes():
    text = "Company <b>EvilCorp</b> reported profits."
    entities = [{"text": "<b>EvilCorp</b>", "label": "ORG", "start": 8, "end": 23}]
    html_output = build_safe_entity_html(text, entities)

    # Invariant: Entity text itself containing tags must be safely escaped
    assert "<b>EvilCorp</b>" not in html_output
    assert "&lt;b&gt;EvilCorp&lt;/b&gt;" in html_output
    assert "entity-org" in html_output


def test_build_safe_entity_html_empty_entities():
    text = "Pure plain text without any named entities."
    html_output = build_safe_entity_html(text, [])
    assert "<mark" not in html_output
    assert "Pure plain text without any named entities." in html_output
