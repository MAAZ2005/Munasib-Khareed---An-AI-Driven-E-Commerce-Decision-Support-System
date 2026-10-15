"""Shared fixtures for the Streamlit end-to-end tests.

Every test here runs a real page through ``streamlit.testing.v1.AppTest``, which
executes the whole script - sidebar included. The sidebar reads ``GET /health``,
so an unpatched test would open a socket to ``127.0.0.1:8000`` and pay the API
client's retry/backoff (about 2.4s) whenever no server is listening, which is
enough to blow the AppTest 3s run timeout.

The API client is therefore patched for every test in this package: these tests
never touch the network, and ``streamlit``'s ``cache_data`` is cleared between
tests so one test's fixture can never satisfy another test's assertion.
"""

from __future__ import annotations

import unittest.mock as mock
from datetime import UTC, datetime
from typing import Any

import pytest

from munasib.api.schemas import HealthResponse


def build_health_response() -> HealthResponse:
    """Build the ``GET /health`` payload returned by the patched client.

    Returns:
        A healthy response for the project version with a reachable database.
    """
    return HealthResponse(
        status="ok",
        service="Munasib Khareed",
        version="0.1.0",
        database="reachable",
        time=datetime.now(UTC),
    )


@pytest.fixture(autouse=True)
def api_is_reachable() -> Any:
    """Answer ``GET /health`` with a fixture instead of opening a socket.

    Yields:
        Control back to the test with the patch active.
    """
    with mock.patch(
        "app.api_client.ApiClient.health",
        return_value=build_health_response(),
    ):
        yield


@pytest.fixture(autouse=True)
def _clear_streamlit_cache() -> None:
    """Drop cached API payloads so no test can read another test's fixture."""
    import streamlit as st

    try:
        st.cache_data.clear()
    except Exception:
        pass
