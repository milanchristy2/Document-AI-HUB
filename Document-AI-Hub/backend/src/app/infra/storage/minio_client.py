import logging
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.config.config import settings

logger = logging.getLogger(__name__)


class MinioClient:
    def __init__(self):
        self._client: Optional[Minio] = None
        try:
            secure = bool(settings.MINIO_SECURE)
            self._client = Minio(
                settings.MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=secure,
            )
        except Exception as e:
            logger.warning("MinIO client init failed: %s", e)
            self._client = None

    def ensure_bucket(self, bucket: str) -> bool:
        try:
            if not self._client:
                return False
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
            return True
        except Exception as e:
            logger.debug("ensure_bucket failed: %s", e)
            return False

    def put_object(self, bucket: str, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        # Attempt several times with a small backoff to mitigate transient MinIO/network errors
        from time import sleep
        from io import BytesIO

        if not self._client:
            logger.debug("MinIO client not initialized, cannot put object %s/%s", bucket, object_name)
            return False

        attempts = 3
        delay = 0.2
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                bio = BytesIO(data)
                # Minio expects endpoint without scheme; it's handled during init
                self._client.put_object(bucket, object_name, bio, length=len(data), content_type=content_type)
                return True
            except Exception as e:
                last_exc = e
                logger.debug("MinIO put_object attempt %d failed for %s/%s: %s", i + 1, bucket, object_name, e)
                sleep(delay)
                delay *= 2

        # Log the final error with full context
        logger.exception("MinIO put_object failed after %d attempts for %s/%s: %s", attempts, bucket, object_name, last_exc)
        return False

    def get_object(self, bucket: str, object_name: str) -> Optional[bytes]:
        try:
            if not self._client:
                return None
            resp = self._client.get_object(bucket, object_name)
            data = resp.read()
            resp.close()
            resp.release_conn()
            return data
        except Exception as e:
            logger.debug("MinIO get_object failed: %s", e)
            return None

    def delete_object(self, bucket: str, object_name: str) -> bool:
        try:
            if not self._client:
                return False
            self._client.remove_object(bucket, object_name)
            return True
        except Exception as e:
            logger.debug("MinIO delete_object failed: %s", e)
            return False


minio_client = MinioClient()
