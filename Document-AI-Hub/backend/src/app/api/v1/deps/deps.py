from pathlib import Path
from typing import AsyncGenerator

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config.config import settings
from app.core.security import decode_jwt
from app.infra.db.session import AsyncSessionLocal, init_engine
from sqlalchemy.ext.asyncio import AsyncSession

bearer_scheme = HTTPBearer(auto_error=False)

async def _bearer(creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> HTTPAuthorizationCredentials:
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return creds

async def get_token_payload(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    token = creds.credentials
    try:
        payload = decode_jwt(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
    return payload

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    # ensure engine initialized
    if AsyncSessionLocal is None:
        init_engine()
    if AsyncSessionLocal is None:
        # fallback: try to initialize a local sqlite aiosqlite engine for dev
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
            fallback_db = (Path(__file__).resolve().parents[5] / "dev.db").resolve()
            url = f"sqlite+aiosqlite:///{fallback_db}"
            engine = create_async_engine(url, future=True)
            globals()['AsyncSessionLocal'] = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        except Exception:
            raise HTTPException(status_code=500, detail="Database engine not initialized")

    async with AsyncSessionLocal() as session: #type:ignore
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
