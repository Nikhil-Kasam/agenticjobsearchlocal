"""
db/session.py — Async SQLAlchemy engine, session factory, and DB init.
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from dotenv import load_dotenv
from db.models import Base

load_dotenv()

ASYNC_DATABASE_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://jobagent:jobagent@localhost:5432/jobagent"
)

# ── Async engine (used by FastAPI + Celery async tasks) ───────────────────────
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """Create all tables. Called once on app startup."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  ✓ Database tables created/verified")


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
