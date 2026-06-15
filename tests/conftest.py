import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: end-to-end tests that require Playwright and full pipeline")
