"""
kk_utils.database — Database session management

Provides database context managers for kk-utils services.
Supports both SQLite and PostgreSQL.

Usage:
    from kk_utils.database import get_db_context
    
    with get_db_context() as db:
        db.query(MyModel).all()
"""

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from typing import Generator, Optional
import os

# Database URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./kk_utils.db")

# Create database engine
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,  # Don't log SQL by default
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class for models
Base = declarative_base()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for getting a database session.
    Use in kk-utils services that need database access.

    Usage:
        from kk_utils.database import get_db_context
        
        with get_db_context() as db:
            db.query(MyModel).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables.
    Call this once at application startup.
    """
    Base.metadata.create_all(bind=engine)
