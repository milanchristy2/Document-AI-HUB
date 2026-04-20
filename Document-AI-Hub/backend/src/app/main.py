import asyncio
import inspect
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ensure `backend/src` is on sys.path so running `python3 main.py` from the repo root works
# Define the project's absolute root directory to resolve all paths correctly
ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = ROOT / "backend"
SRC_ROOT = BACKEND_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Provide sane defaults for local runs before the settings object is instantiated
os.environ.setdefault("SECRET_KEY", "dev-secret-key-0123456789abcdef0123")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("USE_SQLITE", "1")
force_sqlite = os.environ.get("USE_SQLITE", "1")
if force_sqlite.lower() in ("1", "true", "yes"):
    fallback_db_path = (BACKEND_ROOT / "dev.db").resolve()
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{fallback_db_path}"

from app.config.config import settings
from app.api.v1.router import v1_router
from app.exceptions.handlers import register_handlers
from app.middleware.rate_limiter import RateLimitMiddleware
from app.infra.db.session import init_engine, dispose_engine, Base as _Base
from sqlalchemy import create_engine as _create_sync_engine
import redis.asyncio as redis

# Import models so they register with Base.metadata (required for create_all)
try:
    from app.models import User, Document, ChatMessage, EvalRecord
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning("Failed to import models: %s", e)
from starlette.staticfiles import StaticFiles

from alembic.config import Config as AlembicConfig
from alembic import command as alembic_command

logger = logging.getLogger(__name__)


def _sync_database_url(url: str | None) -> str:
    if not url:
        return ""
    return url.replace("+aiosqlite", "").replace("+asyncpg", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    alembic_cfg = None
    alembic_success = False
    sync_db_url = _sync_database_url(settings.DATABASE_URL)
    repo_root = Path(__file__).resolve().parents[3]
    alembic_ini_path = repo_root / "alembic.ini"
    fallback_sync_url = str((BACKEND_ROOT / "dev.db").resolve())

    if alembic_ini_path.exists():
        alembic_cfg = AlembicConfig(str(alembic_ini_path))
        alembic_cfg.set_main_option("script_location", str(repo_root / "alembic"))
        target_sync_url = sync_db_url or fallback_sync_url
        alembic_cfg.set_main_option("sqlalchemy.url", target_sync_url)
        try:
            alembic_command.upgrade(alembic_cfg, "head")
            logger.info("Alembic migrations applied successfully on %s", target_sync_url)
            alembic_success = True
        except Exception:
            logger.exception("Alembic migrations failed; falling back to SQLAlchemy create_all")

    if not alembic_success:
        logger.info("Attempting SQLAlchemy create_all fallback.")
        try:
            sync_engine_url = sync_db_url or fallback_sync_url
            sync_engine = _create_sync_engine(sync_engine_url)
            _Base.metadata.create_all(bind=sync_engine)
            sync_engine.dispose()
            if alembic_cfg is not None:
                try:
                    alembic_command.stamp(alembic_cfg, "head")
                except Exception:
                    logger.exception("Failed to stamp Alembic head after create_all fallback")
        except Exception:
            logger.exception("SQLAlchemy create_all fallback failed")

    init_engine()

    try:
        try:
            if settings.REDIS_URL:
                r = redis.from_url(settings.REDIS_URL, decode_responses=True)
                pong = r.ping()
                if inspect.isawaitable(pong):
                    await pong
                logger.info("Redis is reachable.")
                await r.aclose()
        except Exception:
            logger.warning("Redis is not reachable, continuing in fail-open mode.")
        yield
    finally:
        await dispose_engine()
        try:
            if settings.REDIS_URL:
                r = redis.from_url(settings.REDIS_URL, decode_responses=True)
                await r.aclose()
                logger.info("Redis connection closed.")
        except Exception:
            pass
            
# Main FastAPI app instance
app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
# mount backend static assets if present (serves backend/static at /static)
try:
    static_dir = Path(__file__).resolve().parents[2] / 'static'
    if static_dir.exists():
        app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')
except Exception:
    pass
app.add_middleware(CORSMiddleware, allow_origins=settings.ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RateLimitMiddleware)
register_handlers(app)
app.include_router(v1_router)

@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME, "provider": settings.DEFAULT_PROVIDER}


def _get_run_host_port():
    # allow overriding via environment variable PORT or settings
    import os

    port = int(os.environ.get("PORT") or getattr(settings, "PORT", None) or 8000)
    host = os.environ.get("HOST") or getattr(settings, "HOST", "0.0.0.0")
    return host, port


if __name__ == "__main__":
    import uvicorn
    host, port = _get_run_host_port()
    logger.info("Starting server at %s:%s (debug=%s)", host, port, settings.DEBUG)
    # run the FastAPI app instance directly to avoid module import quirks
    uvicorn.run(app, host=host, port=port, reload=bool(settings.DEBUG))
