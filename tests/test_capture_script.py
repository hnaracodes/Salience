"""Tests for capture script URL injection and SSRF re-validation."""
from pathlib import Path
from unittest.mock import patch

import pytest

from services.api.ssrf import SSRFViolation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_capture_script_injects_url_for_explore_profile():
    from scripts.record_website_session import _load_capture_script

    script_path = PROJECT_ROOT / "configs" / "explore_production.yaml"
    script = _load_capture_script(script_path, "https://example.com/")

    assert script["initial_url"] == "https://example.com/"
    assert script["scroll_mode"] == "explore"
    assert script["viewport"]["width"] == 1280


def test_load_capture_script_rejects_private_url():
    from scripts.record_website_session import _load_capture_script

    script_path = PROJECT_ROOT / "configs" / "explore_production.yaml"
    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("192.168.1.1", 80))]):
        with pytest.raises(SystemExit, match="URL rejected"):
            _load_capture_script(script_path, "http://192.168.1.1/")


def test_load_capture_script_requires_url_for_explore_only_yaml():
    from scripts.record_website_session import _load_capture_script

    script_path = PROJECT_ROOT / "configs" / "explore_production.yaml"
    with pytest.raises(SystemExit, match="Missing initial_url"):
        _load_capture_script(script_path, None)
