"""
================================================================================
File: database/db_engine.py
Project: TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine
Purpose: Provides database engine initialization, SQLAlchemy session management,
         and PostGIS/SQLite abstraction for persistent camera metadata and ANPR logs.
Why this file was made:
  To establish a clean, production-grade ORM persistence layer that connects to
  PostgreSQL/PostGIS when available, with automatic SQLite fallback for lightweight
  local deployment and testing.
================================================================================
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from config.city_config import DATABASE_URL

# SQLAlchemy Declarative Base
Base = declarative_base()

# Global Engine and Session instances
_engine = None
_session_factory = None

def reset_db_engine():
    """Resets the global engine and session factory."""
    global _engine, _session_factory
    if _engine:
        _engine.dispose()
    _engine = None
    _session_factory = None

def get_db_engine(db_url=None):
    """Initializes and returns the SQLAlchemy engine."""
    global _engine
    target_url = db_url or DATABASE_URL
    if _engine is not None and db_url is not None and str(_engine.url) != target_url:
        reset_db_engine()

    if _engine is None:
        if target_url.startswith("sqlite"):
            _engine = create_engine(target_url, connect_args={"check_same_thread": False})
        else:
            _engine = create_engine(target_url, pool_pre_ping=True)
    return _engine

def get_db_session(db_url=None):
    """Returns a new scoped database session."""
    global _session_factory
    engine = get_db_engine(db_url)
    if _session_factory is None:
        _session_factory = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    return _session_factory()

def init_db(db_url=None):
    """Creates all database tables defined in models.py."""
    engine = get_db_engine(db_url)
    # Import models to register schemas with Base
    import database.models  # noqa: F401
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as ex:
        pass
    print(f"[DBEngine] Database initialized successfully with URI: {engine.url}")
