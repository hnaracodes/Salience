"""SSRF protection: validate URLs before passing to the Playwright worker.

Blocks:
- Non-http/https schemes (file:, data:, javascript:, ftp:, etc.)
- Credentials in URL (user:pass@host)
- Private/loopback/link-local IP ranges (RFC1918, ::1, 169.254.0.0/16, cloud metadata)
- Re-validates after every redirect (caller must call validate_url on each hop)

Set ``SSRF_ALLOW_LOCALHOST=1`` only in local pytest/integration runs to permit
127.0.0.0/8 fixture servers. Never enable in production API or worker.
"""
import ipaddress
import os
import socket
import urllib.parse
import tldextract


class SSRFViolation(Exception):
    """Raised when a URL fails SSRF validation."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Cloud metadata endpoints that must be blocked even if they happen to
# resolve to a routable-looking address.
_BLOCKED_IPS = {"169.254.169.254", "100.100.100.200"}


def validate_url(url: str) -> str:
    """Return the URL unchanged if it is safe; raise SSRFViolation otherwise."""
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise SSRFViolation(f"Scheme '{parsed.scheme}' is not allowed; use http or https")

    # Credentials embedded in the netloc (user:pass@host) are a common bypass.
    if "@" in parsed.netloc:
        raise SSRFViolation("Credentials in URL are not allowed")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFViolation("URL has no hostname")

    try:
        resolved = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise SSRFViolation(f"Cannot resolve hostname '{hostname}': {exc}") from exc

    for _family, _type, _proto, _canonname, sockaddr in resolved:
        ip_str = sockaddr[0]
        _check_ip(ip_str)

    return url


def _check_ip(ip_str: str) -> None:
    """Raise SSRFViolation if ip_str is in a forbidden range."""
    if ip_str in _BLOCKED_IPS:
        raise SSRFViolation(f"IP {ip_str} is a cloud metadata endpoint")

    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError as exc:
        raise SSRFViolation(f"Cannot parse resolved IP '{ip_str}': {exc}") from exc

    if os.environ.get("SSRF_ALLOW_LOCALHOST") == "1" and addr.is_loopback:
        return

    # Check loopback before private because 127.x.x.x is both in Python 3.11+.
    if addr.is_loopback:
        raise SSRFViolation(f"IP {ip_str} is a loopback address")
    if addr.is_private:
        raise SSRFViolation(f"IP {ip_str} is in a private range")
    if addr.is_link_local:
        raise SSRFViolation(f"IP {ip_str} is a link-local address")
    if addr.is_multicast:
        raise SSRFViolation(f"IP {ip_str} is a multicast address")
    if addr.is_unspecified:
        raise SSRFViolation(f"IP {ip_str} is an unspecified address")


def get_registrable_domain(url: str) -> str:
    """Return the registrable domain (e.g. 'example.com') for allowed-host guards."""
    extracted = tldextract.extract(url)
    return f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
