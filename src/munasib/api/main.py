"""FastAPI application factory for Munasib Khareed.

``create_app()`` builds the application and the module-level ``app`` makes
``uvicorn munasib.api.main:app`` work. The wiring lives here and nowhere else:

* **Lifespan** opens the SQLite serving cache from the settings, stores the
  session factory on ``app.state`` and shuts the engine down on exit. A missing
  database fails startup with a message that names ``scripts/ingest_db.py``
  instead of letting the first request discover it.
* **Request-ID middleware** honours an inbound ``X-Request-ID`` or generates a
  UUID4, echoes it in the response header, and logs method, path, status and
  duration in milliseconds with the same id.
* **Exception handlers** turn the domain exceptions into the shared
  :class:`~munasib.api.schemas.common.ErrorResponse` envelope: product lookups
  become ``404``, request/validation errors ``422`` with field details, model
  errors ``503``, and everything else a generic ``500`` whose real cause is only
  in the log.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from munasib.api.routers import (
    alternatives_router,
    analysis_router,
    health_router,
    meta_router,
    product_router,
    review_router,
)
from munasib.api.schemas.common import ErrorResponse
from munasib.core.config import Settings, load_settings
from munasib.core.exceptions import (
    ConfigError,
    DataError,
    DataNotFoundError,
    ModelError,
    ProductNotFoundError,
)
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
)

#: Header name for the correlation id (inbound and outbound).
REQUEST_ID_HEADER = "X-Request-ID"

#: Logger used by the request middleware and the error handlers.
api_logger = logging.getLogger("munasib.api")

#: Application error envelope used by every failure path.
ERROR_CODE_PRODUCT_NOT_FOUND = "PRODUCT_NOT_FOUND"
ERROR_CODE_VALIDATION = "VALIDATION_ERROR"
ERROR_CODE_MODEL = "MODEL_UNAVAILABLE"
ERROR_CODE_INTERNAL = "INTERNAL_ERROR"


def _validation_detail(errors: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Normalise validation errors into a JSON-safe, client-friendly block.

    FastAPI and pydantic both report ``loc`` as a tuple and may omit ``ctx``; a
    ``url`` key is dropped as noise and non-scalar constraints are stringified,
    so the envelope always serialises.

    Args:
        errors: Raw error mappings from FastAPI or pydantic.

    Returns:
        ``{"errors": [...]}`` with one flat entry per failing field.
    """
    normalised: list[dict[str, object]] = []
    for error in errors:
        location = error.get("loc", ())
        field = ".".join(str(part) for part in location) if isinstance(location, tuple) else ""
        context = error.get("ctx") or {}
        entry: dict[str, object] = {
            "field": field,
            "type": str(error.get("type", "value_error")),
            "message": str(error.get("msg", "")),
        }
        if "input" in error:
            entry["value"] = error["input"]
        if isinstance(context, dict):
            entry["constraint"] = {
                str(key): (value if isinstance(value, int | float | str | bool) else str(value))
                for key, value in context.items()
            }
        normalised.append(entry)
    return {"errors": normalised}


def _request_id(request: Request) -> str | None:
    """Read the request id the middleware stashed on the request state.

    Args:
        request: The incoming request.

    Returns:
        The correlation id, or None when the middleware did not run.
    """
    return getattr(request.state, "request_id", None)


def _error(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    detail: dict[str, object] | None = None,
) -> JSONResponse:
    """Build the shared error envelope.

    Args:
        request: The incoming request (used for the correlation id).
        status_code: HTTP status to return.
        code: Stable machine token such as ``PRODUCT_NOT_FOUND``.
        message: Human-readable summary that is safe to render.
        detail: Optional machine-readable particulars.

    Returns:
        The JSON response carrying the envelope and the ``X-Request-ID`` header.
    """
    payload = ErrorResponse(
        code=code,
        message=message,
        detail=detail,
        request_id=_request_id(request),
    )
    # The X-Request-ID header itself is written by RequestIDMiddleware, which
    # replaces any existing value so the id appears exactly once.
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def _install_error_handlers(app: FastAPI) -> None:
    """Register every domain exception handler on the application.

    Args:
        app: The application being configured.
    """

    @app.exception_handler(ProductNotFoundError)
    async def _product_not_found(request: Request, exc: ProductNotFoundError) -> JSONResponse:
        """Answer 404 for an ASIN that is not in the serving cache."""
        api_logger.info("product not found: %s", exc.details)
        return _error(
            request,
            status_code=404,
            code=ERROR_CODE_PRODUCT_NOT_FOUND,
            message=exc.message,
            detail=exc.details or None,
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Answer 422 with the offending fields of a bad query parameter."""
        return _error(
            request,
            status_code=422,
            code=ERROR_CODE_VALIDATION,
            message="Unknown or out-of-range query parameter value.",
            detail=_validation_detail(exc.errors()),
        )

    @app.exception_handler(PydanticValidationError)
    async def _pydantic_validation(request: Request, exc: PydanticValidationError) -> JSONResponse:
        """Answer 422 with the per-field details of a failed response model."""
        return _error(
            request,
            status_code=422,
            code=ERROR_CODE_VALIDATION,
            message="Response failed schema validation.",
            detail=_validation_detail(exc.errors()),
        )

    @app.exception_handler(ModelError)
    async def _model_error(request: Request, exc: ModelError) -> JSONResponse:
        """Answer 503 for a missing or broken model/service implementation."""
        api_logger.error("model/service error: %s", exc)
        return _error(
            request,
            status_code=503,
            code=ERROR_CODE_MODEL,
            message="The analysis service is not available.",
            detail=exc.details or None,
        )

    @app.exception_handler(ConfigError)
    async def _config_error(request: Request, exc: ConfigError) -> JSONResponse:
        """Answer 500 for a missing or malformed configuration."""
        api_logger.error("configuration error: %s", exc)
        return _error(
            request,
            status_code=500,
            code=ERROR_CODE_INTERNAL,
            message="Internal server error.",
            detail=None,
        )

    @app.exception_handler(DataError)
    async def _data_error(request: Request, exc: DataError) -> JSONResponse:
        """Answer 500 for dataset/parquet/meta failures without leaking details."""
        api_logger.error("data error: %s", exc, exc_info=True)
        return _error(
            request,
            status_code=500,
            code=ERROR_CODE_INTERNAL,
            message="Internal server error.",
            detail=None,
        )

    @app.exception_handler(DataNotFoundError)
    async def _data_not_found(request: Request, exc: DataNotFoundError) -> JSONResponse:
        """Answer 500 for a missing data artefact (subset, meta.json, ...)."""
        api_logger.error("data artefact missing: %s", exc)
        return _error(
            request,
            status_code=500,
            code=ERROR_CODE_INTERNAL,
            message="Internal server error.",
            detail=None,
        )

    @app.exception_handler(SQLAlchemyError)
    async def _sqlalchemy_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        """Answer 500 for database failures, logging the real cause."""
        api_logger.error("database error: %s", exc, exc_info=True)
        return _error(
            request,
            status_code=500,
            code=ERROR_CODE_INTERNAL,
            message="Internal server error.",
            detail=None,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Wrap framework HTTP errors (405, 404 on an unknown path) in the envelope."""
        return _error(
            request,
            status_code=exc.status_code,
            code=ERROR_CODE_VALIDATION
            if exc.status_code == 422
            else (ERROR_CODE_PRODUCT_NOT_FOUND if exc.status_code == 404 else ERROR_CODE_INTERNAL),
            message=str(exc.detail),
            detail=None,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Answer 500 for anything else, never leaking a stack trace."""
        api_logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error(
            request,
            status_code=500,
            code=ERROR_CODE_INTERNAL,
            message="Internal server error.",
            detail=None,
        )


class RequestIDMiddleware:
    """ASGI middleware that correlates every request with a response header.

    An inbound ``X-Request-ID`` is honoured so a dashboard can stitch its own
    trace together; otherwise a UUID4 is generated. The id is echoed in the
    response header, stored on ``request.state`` for the error envelopes, and
    included in the access log line together with the method, path, status and
    the duration in milliseconds.
    """

    def __init__(self, app: Callable[..., Any]) -> None:
        """Keep a reference to the wrapped ASGI application.

        Args:
            app: The next ASGI application in the chain.
        """
        self._app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Handle one ASGI call.

        Args:
            scope: ASGI connection scope.
            receive: Awaitable receiving ASGI messages.
            send: Awaitable sending ASGI messages.
        """
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(REQUEST_ID_HEADER.lower().encode())
        request_id = incoming.decode("latin-1") if incoming else str(uuid.uuid4())

        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        started = perf_counter()
        status_code = 500
        body_started = False

        async def send_wrapper(message: dict[str, Any]) -> None:
            """Annotate the response start with the correlation id and capture status."""
            nonlocal status_code, body_started
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 500))
                raw_headers = [
                    (key, value)
                    for key, value in (message.get("headers") or [])
                    if key != REQUEST_ID_HEADER.lower().encode()
                ]
                raw_headers.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode()))
                message["headers"] = raw_headers
            elif message["type"] == "http.response.body":
                if not body_started:
                    body_started = True
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        finally:
            duration_ms = (perf_counter() - started) * 1000.0
            api_logger.info(
                "%s %s status=%d duration_ms=%.2f request_id=%s",
                scope.get("method", "-"),
                scope.get("path", "-"),
                status_code,
                duration_ms,
                request_id,
            )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        settings: Pre-loaded settings; loaded from config when omitted.

    Returns:
        The configured application, ready to be served by uvicorn.

    Raises:
        RuntimeError: Propagated from the lifespan when the SQLite serving
            cache is missing or unreadable.
    """
    resolved = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        """Open the serving cache, publish the session factory, dispose on exit.

        Args:
            app: The application whose state is initialised.

        Yields:
            Control back to the server once startup validation has passed.

        Raises:
            RuntimeError: If the database file is missing or cannot be opened,
                naming the script that rebuilds it.
        """
        engine = create_engine_from_settings(resolved)
        db_file = Path(resolved.paths.db_path)
        if not db_file.exists():
            engine.dispose()
            raise RuntimeError(
                "SQLite serving cache not found at "
                f"'{db_file}'. Run 'python scripts/ingest_db.py' to build it from "
                "the processed parquet subset."
            )
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except OperationalError as exc:
            engine.dispose()
            raise RuntimeError(
                "SQLite serving cache not found or not readable at "
                f"'{resolved.paths.db_path}'. Run 'python scripts/ingest_db.py' to "
                "build it from the processed parquet subset."
            ) from exc

        app.state.session_factory = create_session_factory(engine)
        app.state.settings = resolved
        app.state.engine = engine
        api_logger.info(
            "serving cache opened at %s (%s v%s)",
            resolved.paths.db_path,
            resolved.project.display_name,
            resolved.project.version,
        )
        try:
            yield
        finally:
            engine.dispose()
            api_logger.info("serving cache closed")

    app = FastAPI(
        title=resolved.project.display_name,
        version=resolved.project.version,
        description=(
            "AI-driven e-commerce decision support: review evidence, fuzzy + "
            "TOPSIS consumer risk and a final verdict for one product."
        ),
        lifespan=lifespan,
    )

    # CORS for the Streamlit frontend; origins come from config, never defaults.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.api.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    _install_error_handlers(app)

    app.include_router(health_router)
    app.include_router(meta_router)
    app.include_router(product_router)
    app.include_router(review_router)
    app.include_router(analysis_router)
    app.include_router(alternatives_router)

    return app


#: Module-level application so ``uvicorn munasib.api.main:app`` works.
app = create_app()


__all__ = ["REQUEST_ID_HEADER", "RequestIDMiddleware", "app", "create_app"]
