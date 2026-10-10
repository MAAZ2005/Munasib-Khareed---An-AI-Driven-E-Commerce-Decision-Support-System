"""Engine/session factory for the SQLite serving cache.

Connection contract applied to **every** connection (verified by tests):

* ``PRAGMA foreign_keys=ON`` - SQLite defaults to OFF per connection, so this
  must be set on connect, not once at create time.
* ``PRAGMA journal_mode=WAL`` - writer/reader concurrency for the serving cache
  (persistent for file databases; a harmless no-op for in-memory databases).
* ``PRAGMA busy_timeout=...`` - wait instead of failing immediately when
  another process holds the write lock.

``session_scope`` is the transaction boundary: it commits on success and rolls
back on any exception, always closing the session. All datetimes written
through this layer are UTC (see :mod:`munasib.db.models`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from munasib.core.config import Settings, get_settings
from munasib.db.models import Base

#: Milliseconds to wait for a write lock before SQLite raises "database is
#: locked" (infrastructural constant, not a tuning threshold).
BUSY_TIMEOUT_MS = 5000

#: SQLAlchemy URLs that mean "in-memory database" (tests).
_MEMORY_URLS = ("sqlite://", "sqlite:///:memory:")


def _apply_pragmas(dbapi_connection: object, _connection_record: object) -> None:
    """Apply the per-connection PRAGMAs for the serving cache.

    Args:
        dbapi_connection: Raw DBAPI connection handed out by the pool.
        _connection_record: Unused (SQLAlchemy connect-event signature).

    Note:
        ``foreign_keys`` is per-connection state in SQLite, so it is set here
        on every connect; ``journal_mode=WAL`` is persistent for file
        databases but setting it per connection is idempotent and also covers
        in-memory databases (which report ``memory`` instead).
    """
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()


def create_engine_from_settings(
    settings: Settings | None = None,
    *,
    db_url: str | None = None,
) -> Engine:
    """Create a SQLAlchemy engine for the serving cache.

    Args:
        settings: Project settings; loaded from config only when ``db_url``
            is not given (tests pass an explicit URL and never touch config).
        db_url: Explicit SQLAlchemy URL (e.g. ``"sqlite://"`` for in-memory).
            When omitted, ``sqlite:///`` + ``settings.paths.db_path`` is used,
            so the database path always comes from config.

    Returns:
        Engine with the PRAGMA listener attached.
    """
    if db_url is None:
        settings = settings or get_settings()
        db_url = f"sqlite:///{settings.paths.db_path}"

    connect_args: dict[str, object] = {}
    engine_kwargs: dict[str, object] = {"future": True}
    if db_url in _MEMORY_URLS:
        # One shared connection keeps an in-memory database alive across sessions.
        connect_args["check_same_thread"] = False
        engine_kwargs["poolclass"] = StaticPool

    engine = create_engine(db_url, connect_args=connect_args, **engine_kwargs)
    event.listen(engine, "connect", _apply_pragmas)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory bound to ``engine``.

    ``expire_on_commit=False`` keeps attributes readable after the session is
    closed, which matters because repositories return ORM objects to callers.

    Args:
        engine: Engine to bind.

    Returns:
        Configured ``sessionmaker``.
    """
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transaction scope: commit on success, roll back on error, always close.

    Args:
        factory: Session factory from :func:`create_session_factory`.

    Yields:
        The active session.

    Raises:
        Exception: Re-rolled after ``rollback()`` so callers see the original
            error (integrity violations, etc.).
    """
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine) -> None:
    """Create every table defined on :class:`munasib.db.models.Base`.

    Existing tables are left untouched (``create_all`` semantics), so this is
    safe to call on every run.

    Args:
        engine: Target engine.
    """
    Base.metadata.create_all(engine)
