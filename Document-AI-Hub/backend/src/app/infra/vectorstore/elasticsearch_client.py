import logging
from typing import Optional
import warnings

from elasticsearch import AsyncElasticsearch
from urllib.parse import urlparse

from app.config.config import settings

logger = logging.getLogger(__name__)

# Suppress Elasticsearch security warning for local development
# In production, enable Elasticsearch security: https://www.elastic.co/guide/en/elasticsearch/reference/7.17/security-minimal-setup.html
warnings.filterwarnings("ignore", message="Elasticsearch built-in security features are not enabled")

_es: Optional[AsyncElasticsearch] = None


def _is_valid_url(u: str | None) -> bool:
    if not u:
        return False
    try:
        p = urlparse(u)
        return bool(p.scheme and p.hostname and p.port)
    except Exception:
        return False


def get_es_client(retries: int = 1, backoff_seconds: float = 0.5) -> Optional[AsyncElasticsearch]:
    """Return a cached AsyncElasticsearch client or None if not configured/available.

    This function avoids initializing the client when `settings.ELASTICSEARCH_URL` is empty
    or malformed. On transient failures it will retry a small number of times.
    """
    global _es
    if _es is not None:
        return _es

    url = settings.ELASTICSEARCH_URL
    if not _is_valid_url(url):
        logger.debug("Elasticsearch URL not configured or invalid: %s", url)
        return None

    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            _es = AsyncElasticsearch(hosts=[url])
            return _es
        except Exception as e:
            last_exc = e
            logger.warning("Elasticsearch init attempt %d failed: %s", attempt + 1, e)
            try:
                import time

                time.sleep(backoff_seconds * (attempt + 1))
            except Exception:
                pass

    logger.warning("Elasticsearch init failed after %d attempts: %s", retries, last_exc)
    _es = None
    return None


async def close_es():
    global _es
    if _es:
        try:
            await _es.close()
        except Exception:
            pass
        _es = None
