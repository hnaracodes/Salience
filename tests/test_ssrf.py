"""Unit tests for the SSRF gate (services/api/ssrf.py).

DNS resolution is mocked so tests run without network access.
"""
from unittest.mock import patch

import pytest

from services.api.ssrf import SSRFViolation, validate_url

# Mock return for getaddrinfo: (family, type, proto, canonname, sockaddr)
def _mock_dns(ip: str):
    """Return a getaddrinfo-style response resolving to the given IP."""
    return [(2, 1, 6, "", (ip, 80))]


def test_rejects_private_ip():
    with patch("socket.getaddrinfo", return_value=_mock_dns("192.168.1.1")):
        with pytest.raises(SSRFViolation, match="private"):
            validate_url("http://192.168.1.1/admin")


def test_rejects_metadata_endpoint():
    with patch("socket.getaddrinfo", return_value=_mock_dns("169.254.169.254")):
        with pytest.raises(SSRFViolation, match="metadata"):
            validate_url("http://169.254.169.254/latest/meta-data/")


def test_rejects_localhost():
    with patch("socket.getaddrinfo", return_value=_mock_dns("127.0.0.1")):
        with pytest.raises(SSRFViolation, match="loopback"):
            validate_url("http://localhost/secret")


def test_rejects_file_scheme():
    # No DNS call needed — scheme check happens first.
    with pytest.raises(SSRFViolation, match="Scheme"):
        validate_url("file:///etc/passwd")


def test_accepts_public_url():
    with patch("socket.getaddrinfo", return_value=_mock_dns("93.184.216.34")):
        result = validate_url("https://example.com")
    assert result == "https://example.com"


def test_rejects_credentials_in_url():
    with pytest.raises(SSRFViolation, match="Credentials"):
        validate_url("http://user:pass@example.com")


def test_allow_localhost_env_permits_loopback_only(monkeypatch):
    monkeypatch.setenv("SSRF_ALLOW_LOCALHOST", "1")
    with patch("socket.getaddrinfo", return_value=_mock_dns("127.0.0.1")):
        assert validate_url("http://127.0.0.1:8765/index.html") == "http://127.0.0.1:8765/index.html"


def test_allow_localhost_env_still_blocks_private(monkeypatch):
    monkeypatch.setenv("SSRF_ALLOW_LOCALHOST", "1")
    with patch("socket.getaddrinfo", return_value=_mock_dns("192.168.1.1")):
        with pytest.raises(SSRFViolation, match="private"):
            validate_url("http://192.168.1.1/")
