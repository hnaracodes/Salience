"""API security smoke tests (auth gate, input validation)."""
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from services.api.auth import get_current_user
from services.api.schemas import ScanCreate


def test_scan_create_rejects_oversized_site_goal():
    with pytest.raises(ValidationError):
        ScanCreate(url="https://example.com/", site_goal="x" * 2001)


def test_get_current_user_fails_closed_when_clerk_not_configured():
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="fake-token")
    with patch.dict(os.environ, {"CLERK_JWKS_URL": "", "CLERK_ISSUER": ""}, clear=False):
        with patch("services.api.auth.CLERK_JWKS_URL", ""):
            with patch("services.api.auth.CLERK_ISSUER", ""):
                with pytest.raises(HTTPException) as exc_info:
                    get_current_user(creds)
                assert exc_info.value.status_code == 503
