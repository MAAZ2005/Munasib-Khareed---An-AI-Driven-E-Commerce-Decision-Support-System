"""FastAPI dependencies shared by every router.

Three dependency families live here so no router hand-writes plumbing:

* **Session** - :func:`get_session` yields a session built from the factory the
  lifespan stored on ``app.state`` and always closes it.
* **Settings** - :func:`get_settings` returns the cached project settings.
* **Repositories / service** - thin constructors over the session, plus
  :func:`get_analysis_service`, the single place that chooses between the stub
  and the real engine.

The service switch is ``analysis.service`` in ``configs/config.yaml``
(``stub`` by default, overridable with ``MUNASIB_ANALYSIS_SERVICE``). Selecting
``real`` raises :class:`~munasib.core.exceptions.ModelNotLoadedError` - a
``ModelError``, so the API answers ``503`` with the standard error envelope -
because the fuzzy + TOPSIS engine does not exist yet. It never silently falls
back to the stub, which would present placeholder numbers as a real result.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from munasib.core.config import Settings
from munasib.core.config import get_settings as load_settings
from munasib.core.exceptions import ModelNotLoadedError
from munasib.db.repositories import (
    ProductRepository,
    RelatedRepository,
    ReviewRepository,
)
from munasib.pipeline.service import AnalysisService, StubAnalysisService

#: Message used when ``analysis.service: real`` is selected before the engine
#: exists. Kept here so the error text and the documented cause stay together.
REAL_SERVICE_ERROR = (
    'analysis.service is "real", but the fuzzy + TOPSIS + SHAP engine is not '
    "implemented yet. Set analysis.service to \"stub\" in configs/config.yaml "
    "(or export MUNASIB_ANALYSIS_SERVICE=stub) until the engine lands; the stub "
    "is the only implementation available today."
)


def get_settings(request: Request) -> Settings:
    """Return the settings this application was built with.

    The app's own settings are preferred so a caller that constructs
    ``create_app(custom_settings)`` (tests, embedded deployments) is honoured
    everywhere; the cached global loader is only the fallback when the lifespan
    never ran.

    Args:
        request: The incoming request, used to read ``app.state.settings``.

    Returns:
        The project settings.
    """
    app_settings = getattr(request.app.state, "settings", None)
    if isinstance(app_settings, Settings):
        return app_settings
    return load_settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_session(request: Request) -> Iterator[Session]:
    """Yield a session bound to the engine created by the app lifespan.

    Args:
        request: The incoming request, used to read ``app.state.session_factory``.

    Yields:
        An open session; closed in the ``finally`` block so a failing route can
        never leak a connection.

    Raises:
        RuntimeError: If the lifespan never ran (the server was started without
            a usable database).
    """
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise RuntimeError(
            "no database session factory on app.state; the lifespan did not open "
            "the SQLite serving cache (is data/db/munasib.sqlite missing? run "
            "python scripts/ingest_db.py)"
        )
    session = factory()
    try:
        yield session
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


def get_product_repository(session: SessionDep) -> ProductRepository:
    """Build the product repository for this request.

    Args:
        session: Request-scoped session.

    Returns:
        The product repository.
    """
    return ProductRepository(session)


def get_review_repository(session: SessionDep) -> ReviewRepository:
    """Build the review repository for this request.

    Args:
        session: Request-scoped session.

    Returns:
        The review repository.
    """
    return ReviewRepository(session)


def get_related_repository(session: SessionDep) -> RelatedRepository:
    """Build the related-product repository for this request.

    Args:
        session: Request-scoped session.

    Returns:
        The related-product repository.
    """
    return RelatedRepository(session)


ProductRepoDep = Annotated[ProductRepository, Depends(get_product_repository)]
ReviewRepoDep = Annotated[ReviewRepository, Depends(get_review_repository)]
RelatedRepoDep = Annotated[RelatedRepository, Depends(get_related_repository)]


def get_analysis_service(session: SessionDep, settings: SettingsDep) -> AnalysisService:
    """Return the analysis implementation selected by the configuration.

    Args:
        session: Request-scoped session.
        settings: Project settings (``analysis.service``).

    Returns:
        The stub service for ``stub``.

    Raises:
        ModelNotLoadedError: If ``analysis.service`` is ``real``, which has no
            implementation yet. The API surfaces this as ``503`` rather than
            quietly serving placeholder numbers.
    """
    if settings.analysis.service == "stub":
        return StubAnalysisService(session, settings)
    raise ModelNotLoadedError(REAL_SERVICE_ERROR, details={"service": "real"})


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]


__all__ = [
    "AnalysisServiceDep",
    "ProductRepoDep",
    "REAL_SERVICE_ERROR",
    "RelatedRepoDep",
    "ReviewRepoDep",
    "SessionDep",
    "SettingsDep",
    "get_analysis_service",
    "get_product_repository",
    "get_related_repository",
    "get_review_repository",
    "get_session",
    "get_settings",
]
