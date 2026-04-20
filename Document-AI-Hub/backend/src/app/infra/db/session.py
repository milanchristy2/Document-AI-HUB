import logging
from typing import AsyncGenerator
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine = None
AsyncSessionLocal = None

# create async engine only when an async driver is present in the URL
def init_engine():
    global _engine, AsyncSessionLocal
    url = settings.DATABASE_URL

    # For local sqlite, prefer the async aiosqlite driver to keep session
    # code path consistent.  If the URL is plain sqlite, normalize it.
    if url.startswith("sqlite://") and "+aiosqlite" not in url:
        async_url = url.replace("sqlite:///", "sqlite+aiosqlite:///")
        settings.DATABASE_URL = async_url
    else:
        async_url = url

    # If an async driver is present in the URL, create an async engine.
    # Otherwise, for common local dev SQLite URLs, create the sync tables
    # so that `create_all` ensures the schema exists.
    if "+asyncpg" in async_url or "+aiosqlite" in async_url:
        # Attempt to prefer a Postgres async engine when configured. If the
        # database is unreachable (network, auth, etc.), fall back to the
        # repository `dev.db` sqlite file and create tables there so local
        # development continues without requiring a running Postgres.
        #
        # For Postgres we try a quick sync connection test (strip +asyncpg)
        # to verify reachability before constructing the async engine.
        try:
            if "+asyncpg" in async_url:
                # build a sync URL by removing the async driver suffix
                sync_url = async_url.replace("+asyncpg", "")
                from sqlalchemy import create_engine as _create_sync_engine

                try:
                    sync_eng = _create_sync_engine(sync_url)
                    conn = sync_eng.connect()
                    conn.close()
                    sync_eng.dispose()
                    # reachable: create async engine with pooling
                    _engine = create_async_engine(
                        async_url,
                        pool_size=10,
                        max_overflow=20,
                        pool_pre_ping=True,
                        future=True,
                    )
                    AsyncSessionLocal = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
                except Exception as e:
                    logger.warning("Postgres test connection failed, falling back to sqlite dev DB: %s", e)
                    # leave _engine/AsyncSessionLocal as None so we create sqlite below
                    _engine = None
                    AsyncSessionLocal = None
            else:
                # aiosqlite specified: use it directly
                _engine = create_async_engine(async_url, future=True)
                AsyncSessionLocal = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
        except Exception:
            logger.exception("Failed to initialize async engine; will fall back to sqlite dev DB")
            _engine = None
            AsyncSessionLocal = None
    # If we reached here and no async engine/sessionmaker was created
    # (for example Postgres was configured but unreachable), ensure a
    # local sqlite `dev.db` exists and that models' tables are created so
    # the app can operate in dev mode without Postgres.
    try:
        from sqlalchemy import create_engine as _create_sync_engine
        db_url = url
        if not db_url or ("+asyncpg" in db_url and AsyncSessionLocal is None):
            try:
                repo_root = Path(__file__).resolve().parents[2]
                db_path = (repo_root / "dev.db").resolve()
                db_url = f"sqlite:///{db_path}"
            except Exception:
                pass

        if db_url and db_url.startswith("sqlite"):
            # Persist the fallback in settings so dependent code sees it.
            settings.DATABASE_URL = db_url
            logger.info("Falling back to sqlite database: %s", db_url)

            # For async sessions, prefer sqlite+aiosqlite URL if missing.
            if "+aiosqlite" not in settings.DATABASE_URL:
                settings.DATABASE_URL = settings.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")

            sync_engine = _create_sync_engine(db_url, connect_args={"check_same_thread": False})
            try:
                import importlib
                import pkgutil
                import app.models
                for _, modname, _ in pkgutil.iter_modules(app.models.__path__):
                    importlib.import_module(f"app.models.{modname}")
            except Exception:
                pass
            try:
                Base.metadata.create_all(bind=sync_engine)
            except Exception:
                pass
    except Exception:
        logger.exception("Failed to create sqlite dev.db schema")
    # ensure engine/session variables are in a deterministic state
    if _engine is None:
        AsyncSessionLocal = None


async def dispose_engine():
    global _engine
    if _engine is not None:
        try:
            await _engine.dispose()
        except Exception:
            try:
                _engine.sync_engine.dispose()
            except Exception:
                pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session: #type:ignore
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
