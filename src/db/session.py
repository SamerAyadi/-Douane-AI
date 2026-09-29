import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import DATABASE_URL


logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None

if DATABASE_URL:
    try:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    except Exception as error:
        logger.warning("Database persistence is disabled: %s", error)
else:
    logger.warning(
        "DATABASE_URL is not configured; chat persistence is disabled."
    )


def get_engine() -> Engine | None:
    return _engine


def is_database_configured() -> bool:
    return _session_factory is not None


@contextmanager
def database_session() -> Iterator[Session | None]:
    if _session_factory is None:
        yield None
        return

    database = _session_factory()
    try:
        yield database
    finally:
        database.close()
