from logging.config import fileConfig
import os
import sys
from pathlib import Path
from sqlalchemy import pool, create_engine, NullPool
from sqlalchemy.exc import OperationalError
from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'src')))

force_sqlite = os.environ.get("USE_SQLITE", "1")
if force_sqlite.lower() in ("1", "true", "yes"):
    sqlite_path = (Path(__file__).resolve().parents[1] / "backend" / "dev.db").resolve()
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{sqlite_path}"

from app.infra.db.session import Base #type: ignore
from app.models import *  # noqa: F401,F403 #type: ignore
from app.config import settings #type: ignore


def _sync_database_url(db_url: str | None) -> str:
    if not db_url:
        return ""
    return db_url.replace("+aiosqlite", "").replace("+asyncpg", "")


def resolve_database_url() -> str:
    db_url = settings.DATABASE_URL
    sync_url = _sync_database_url(db_url)

    if sync_url:
        try:
            # Use NullPool so we do not hold a locked connection on SQLite
            engine = create_engine(sync_url, poolclass=NullPool)
            with engine.connect():
                pass
            return sync_url
        except OperationalError:
            pass
        except Exception:
            pass

    # put local fallback dev.db relative to repo root
    repo_root = Path(__file__).resolve().parents[1]
    sqlite_path = (repo_root / "backend" / "dev.db").resolve()
    fallback_url = f"sqlite:///{sqlite_path}"
    return fallback_url


target_metadata = Base.metadata


def run_migrations_offline():
    url = resolve_database_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    from sqlalchemy import engine_from_config

    connectable = engine_from_config(
        {"sqlalchemy.url": resolve_database_url()}, prefix="sqlalchemy.", poolclass=pool.NullPool
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
