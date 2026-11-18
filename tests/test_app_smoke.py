"""Smoke test for application entry point."""

import app


def test_app_entrypoint_structure():
    assert hasattr(app, "main")
    assert hasattr(app, "load_custom_css")
    assert callable(app.main)
    assert callable(app.load_custom_css)
