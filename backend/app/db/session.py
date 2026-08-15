"""SQLAlchemy engine and session management."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


if not settings.DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured. Set it in .env or through Docker Compose."
    )


class Base(DeclarativeBase):
    """Base class for application ORM models."""


engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Generator[Session, None, None]:
    """Provide one database session per request or background operation."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
