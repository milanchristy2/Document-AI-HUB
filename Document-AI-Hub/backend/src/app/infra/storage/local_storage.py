import aiofiles
import asyncio
import logging
import os
import uuid
from typing import Optional

from app.config.config import settings

logger = logging.getLogger(__name__)


class LocalStorage:
    def __init__(self, base_path: str | None = None):
        # If MinIO is configured and available, LocalStorage will use MinIO instead of local FS.
        from app.infra.storage.minio_client import minio_client

        self._use_minio = False
        if getattr(settings, "MINIO_ENDPOINT", None) and getattr(minio_client, "_client", None):
            try:
                bucket = getattr(settings, "MINIO_BUCKET", "uploads") or "uploads"
                bucket_ok = minio_client.ensure_bucket(bucket)
                if bucket_ok:
                    self._minio_bucket = bucket
                    self._use_minio = True
                else:
                    logger.warning("MinIO bucket '%s' could not be ensured, falling back to local FS", bucket)
                    self._minio_bucket = bucket
                    self._use_minio = False
            except Exception as e:
                logger.warning("MinIO bucket init failed for '%s': %s; falling back to local FS", bucket, e)
                self._use_minio = False

        self.base = base_path or settings.UPLOAD_DIR
        os.makedirs(self.base, exist_ok=True)

        self._minio_client = minio_client if self._use_minio else None

    async def save(self, data: bytes, ext: str) -> str:
        fname = f"{uuid.uuid4().hex}.{ext.lstrip('.')}"
        if self._use_minio and self._minio_client:
            try:
                ok = self._minio_client.put_object(self._minio_bucket, fname, data, content_type="application/octet-stream")
                if ok:
                    return fname
                # If put_object returns False (no exception), raise so fallback can happen.
                raise IOError("MinIO put_object returned False")
            except Exception as e:
                logger.exception("MinIO put_object failed for bucket=%s object=%s: %s", self._minio_bucket, fname, e)
                # Fallback to local file storage when MinIO is temporarily unavailable.
                logger.warning("Falling back to local disk storage for %s", fname)
                self._use_minio = False

        path = os.path.join(self.base, fname)
        try:
            async with aiofiles.open(path, "wb") as f:
                await f.write(data)
            return fname
        except Exception as e:
            logger.exception("Failed to save file: %s", e)
            # If MinIO was enabled and local fallback failed, preserve exception trace.
            raise

        path = os.path.join(self.base, fname)
        try:
            async with aiofiles.open(path, "wb") as f:
                await f.write(data)
            return fname
        except Exception as e:
            logger.exception("Failed to save file: %s", e)
            raise

    async def load(self, filename: str) -> Optional[bytes]:
        if self._use_minio and self._minio_client:
            try:
                return self._minio_client.get_object(self._minio_bucket, filename)
            except Exception:
                return None

        path = os.path.join(self.base, filename)
        try:
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        except Exception as e:
            logger.debug("Load failed: %s", e)
            return None

    async def delete(self, filename: str) -> None:
        if self._use_minio and self._minio_client:
            try:
                self._minio_client.delete_object(self._minio_bucket, filename)
                return
            except Exception:
                logger.debug("Failed to delete from minio: %s", filename)

        path = os.path.join(self.base, filename)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.debug("Failed to delete: %s", filename)


local_storage = LocalStorage()
