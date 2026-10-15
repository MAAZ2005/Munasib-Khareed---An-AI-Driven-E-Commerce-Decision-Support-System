"""``GET /health`` - liveness plus reachability of the SQLite serving cache.

The route answers ``200`` whenever the process is serving, even when the cache
is unreachable: ``database`` is the field a dashboard should watch, not the HTTP
status. A missing cache is an operational condition, not a crash.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from sqlalchemy import text

from munasib.api.deps import SessionDep, SettingsDep
from munasib.api.schemas.common import HealthResponse

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthResponse)
def health(session: SessionDep, settings: SettingsDep) -> HealthResponse:
    """Report whether the API and its serving cache are answering.

    Args:
        session: Request-scoped session over the serving cache.
        settings: Project settings (identity fields).

    Returns:
        The health payload with the current UTC time.
    """
    try:
        session.execute(text("SELECT 1"))
        database = "reachable"
    except Exception:  # noqa: BLE001 - any driver failure means "unreachable"
        database = "unavailable"
    return HealthResponse(
        status="ok",
        service=settings.project.display_name,
        version=settings.project.version,
        database=database,
        time=datetime.now(UTC),
    )
