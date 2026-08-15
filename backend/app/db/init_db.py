"""Database schema and initial data setup."""

import logging

from sqlalchemy import text

from app.core.config import settings
from app.db.session import Base, SessionLocal, engine
from app.models.document import DataSource


logger = logging.getLogger(__name__)


def init_db() -> None:
    """Enable pgvector and create all application-owned tables."""
    # Importing the models module above registers every ORM table with Base.metadata.
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)
    logger.info("Database extension and application tables are ready")


def init_default_datasources() -> None:
    """Create the built-in local upload source once."""
    default_name = "Local Uploads"

    with SessionLocal() as db:
        existing = db.query(DataSource).filter(DataSource.name == default_name).first()
        if existing:
            return

        db.add(
            DataSource(
                name=default_name,
                source_type="file_system",
                connection_details={
                    "path": settings.UPLOAD_DIR,
                    "managed": True,
                },
                is_active=True,
            )
        )
        db.commit()
        logger.info("Created default data source: %s", default_name)
