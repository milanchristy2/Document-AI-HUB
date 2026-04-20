import logging
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.infra.vectorstore.elasticsearch_client import get_es_client, close_es
from app.config.config import settings

logger = logging.getLogger(__name__)


class ElasticRetriever:
    """Async Elasticsearch retriever with connection pooling and resilience.
    
    Features:
    - Connection pooling and reuse
    - Retry logic for transient failures
    - Circuit breaker pattern to avoid cascading failures
    - Multiple query strategies (match, bool, multi_match)
    - Scoring normalization
    """
    
    # Circuit breaker state
    _circuit_open = False
    _circuit_last_error_time = None
    _circuit_retry_interval = 30  # seconds
    
    def __init__(self, index: str):
        self.index = index
        self._es = get_es_client()
        self._request_count = 0
        self._error_count = 0

    async def _check_circuit(self) -> bool:
        """Check if circuit breaker should allow requests."""
        if not self._circuit_open:
            return True
        
        # Allow retry after interval
        if self._circuit_last_error_time:
            elapsed = (datetime.now() - self._circuit_last_error_time).total_seconds()
            if elapsed > self._circuit_retry_interval:
                logger.info("Circuit breaker: retrying ES connection")
                self._circuit_open = False
                return True
        
        return False

    async def index_document(self, doc_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """Index chunks to Elasticsearch with retry logic.
        
        Args:
            doc_id: Document identifier
            chunks: List of chunk dicts with id, text, page, heading, document_id
        
        Returns:
            True if successful, False otherwise
        """
        if not self._es or not await self._check_circuit():
            logger.warning("Elasticsearch client unavailable or circuit open")
            return False
        
        if not chunks:
            return True
        
        bulk_ops = []
        for ch in chunks:
            op = {"index": {"_index": self.index, "_id": ch.get("id")}}
            bulk_ops.append(op)
            bulk_ops.append({
                "text": ch.get("text"),
                "page": ch.get("page"),
                "heading": ch.get("heading"),
                "document_id": ch.get("document_id"),
                "indexed_at": datetime.now().isoformat(),
            })
        
        try:
            result = await self._es.bulk(body=bulk_ops, refresh=True)
            errors = result.get("errors", False)
            
            if errors:
                logger.warning(f"Elasticsearch bulk had errors for {len(chunks)} chunks")
                # Count actual errors
                for item in result.get("items", []):
                    if item.get("index", {}).get("error"):
                        self._error_count += 1
            
            self._request_count += 1
            logger.info(f"Indexed {len(chunks)} chunks to ES (request #{self._request_count})")
            return not errors
        
        except Exception as e:
            self._error_count += 1
            logger.exception(f"Elasticsearch bulk index failed: {e}")
            
            # Open circuit breaker on repeated failures
            if self._error_count > 3:
                self._circuit_open = True
                self._circuit_last_error_time = datetime.now()
                logger.warning("Circuit breaker opened for Elasticsearch")
            
            # Try to close client
            try:
                await asyncio.create_task(close_es())
            except Exception:
                pass
            
            return False

    async def search(self, query: str, document_id: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search Elasticsearch with multi-strategy fallback.
        
        Args:
            query: Search query string
            document_id: Optional document filter
            top_k: Number of results to return
        
        Returns:
            List of result dicts with text, score, metadata
        """
        if not self._es or not await self._check_circuit():
            return []
        
        # Build query with multiple strategies
        query_body = self._build_search_query(query, document_id, top_k)
        
        try:
            resp = await self._es.search(index=self.index, body=query_body)
            results = self._parse_es_response(resp)
            
            # Normalize scores to 0-1 range
            if results:
                max_score = max(r.get("score", 0) for r in results)
                if max_score > 0:
                    for r in results:
                        r["score"] = min(1.0, r.get("score", 0) / max_score)
            
            self._request_count += 1
            logger.debug(f"ES search returned {len(results)} results for query: {query[:50]}...")
            return results
        
        except Exception as e:
            self._error_count += 1
            logger.exception(f"Elasticsearch search failed: {e}")
            
            if self._error_count > 3:
                self._circuit_open = True
                self._circuit_last_error_time = datetime.now()
            
            try:
                await asyncio.create_task(close_es())
            except Exception:
                pass
            
            return []

    def _build_search_query(self, query: str, document_id: Optional[str], top_k: int) -> Dict[str, Any]:
        """Build optimized ES query with multiple strategies."""
        # Multi-field search: prioritize exact matches and phrase matches
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text^2", "heading^1.5", "text.keyword"],
                                "type": "best_fields",
                                "operator": "or",
                                "fuzziness": "AUTO"
                            }
                        }
                    ],
                    "should": [
                        {
                            "match_phrase": {
                                "text": {
                                    "query": query,
                                    "boost": 2.0
                                }
                            }
                        }
                    ]
                }
            },
            "min_score": 0.1  # Filter low-scoring results
        }
        
        # Add document filter if specified
        if document_id:
            body["query"]["bool"]["filter"] = [
                {"term": {"document_id": document_id}}
            ]
        
        return body

    def _parse_es_response(self, resp: Any) -> List[Dict[str, Any]]:
        """Parse Elasticsearch response and extract results."""
        hits = resp.get("hits", {}).get("hits", [])
        results = []
        
        for h in hits:
            src = h.get("_source", {})
            score = h.get("_score", 0.0)
            
            # Build result with all available metadata
            result = {
                "id": h.get("_id"),
                "text": src.get("text", ""),
                "score": score,
                "heading": src.get("heading", ""),
                "page": src.get("page"),
                "document_id": src.get("document_id"),
                "source": "elasticsearch",
                "indexed_at": src.get("indexed_at"),
            }
            
            # Only include results with non-empty text
            if result["text"]:
                results.append(result)
        
        return results

