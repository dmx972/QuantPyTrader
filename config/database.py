"""
Database configuration and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool
from config.settings import settings

# SQLAlchemy setup
if settings.database_url.startswith("sqlite"):
    # SQLite specific configuration
    engine = create_engine(
        settings.database_url,
        connect_args={
            "check_same_thread": False,
            "timeout": 30
        },
        poolclass=StaticPool,
        echo=settings.debug
    )
else:
    # For other databases (PostgreSQL, MySQL, etc.)
    engine = create_engine(
        settings.database_url,
        echo=settings.debug
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Session:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    from core.database.models import Base as ModelsBase
    # Import all models to register them with SQLAlchemy
    from core.database import models
    ModelsBase.metadata.create_all(bind=engine)