"""Validate Clerk-issued JWTs.

Uses PyJWT + Clerk JWKS endpoint. JWKS are cached with a 1-hour TTL.
"""
import os
import time
from typing import Any

import httpx
import jwt  # PyJWT
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

CLERK_JWKS_URL = os.environ.get("CLERK_JWKS_URL", "")
CLERK_ISSUER = os.environ.get("CLERK_ISSUER", "")

_bearer_scheme = HTTPBearer(auto_error=True)

# Simple in-process JWKS cache: {"keys": [...], "fetched_at": float}
_jwks_cache: dict[str, Any] = {}
_JWKS_TTL = 3600  # seconds


def get_jwks() -> list[dict]:
    """Return JWKS keys, re-fetching from Clerk if the cache is stale."""
    now = time.monotonic()
    if _jwks_cache and (now - _jwks_cache["fetched_at"]) < _JWKS_TTL:
        return _jwks_cache["keys"]

    response = httpx.get(CLERK_JWKS_URL, timeout=10)
    response.raise_for_status()
    keys = response.json().get("keys", [])
    _jwks_cache["keys"] = keys
    _jwks_cache["fetched_at"] = now
    return keys


def verify_clerk_token(token: str) -> dict:
    """Decode and verify a Clerk JWT. Returns the decoded claims dict."""
    try:
        unverified_header = jwt.get_unverified_header(token)
    except jwt.DecodeError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token header: {exc}") from exc

    kid = unverified_header.get("kid")
    keys = get_jwks()
    matching = [k for k in keys if k.get("kid") == kid]
    if not matching:
        # Key might have rotated; bust cache and retry once.
        _jwks_cache.clear()
        keys = get_jwks()
        matching = [k for k in keys if k.get("kid") == kid]

    if not matching:
        raise HTTPException(status_code=401, detail="No matching JWKS key found for token kid")

    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(matching[0])

    try:
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
            options={"verify_exp": True},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token has expired") from exc
    except jwt.InvalidIssuerError as exc:
        raise HTTPException(status_code=401, detail="Token issuer is invalid") from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {exc}") from exc

    return claims


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme),
) -> dict:
    """FastAPI dependency: validate Bearer token and return Clerk claims."""
    if not CLERK_JWKS_URL or not CLERK_ISSUER:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    return verify_clerk_token(credentials.credentials)
