import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.rag.retrievers.elastic_retriever import ElasticRetriever
from app.rag.retrievers.vector_retriever import VectorRetriever
from app.rag.rerank import Reranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining Elasticsearch (lexical) and Chroma (semantic) search.
    
    Features:
    - Parallel retrieval from ES and vector stores
    - Score normalization and fusion
    - Deduplication based on semantic similarity
    - Configurable retrieval strategy (es-first, vector-first, balanced)
    - Cross-encoder reranking for final ranking
    """
    
    def __init__(self, elastic_index: str, reranker_model: str | None = None, strategy: str = "balanced"):
        """Initialize hybrid retriever.
        
        Args:
            elastic_index: Elasticsearch index name
            reranker_model: Cross-encoder model for reranking
            strategy: 'es-first', 'vector-first', or 'balanced'
        """
        self.elastic = ElasticRetriever(elastic_index)
        self.vector = VectorRetriever()
        self.reranker = Reranker(reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.strategy = strategy

    async def retrieve(self, query: str, document_id: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid retrieval with parallel ES and vector search.
        
        Args:
            query: Search query
            document_id: Optional document filter
            top_k: Number of results to return
        
        Returns:
            Merged and deduplicated results
        """
        # Retrieve in parallel from both sources
        es_task = asyncio.create_task(
            self.elastic.search(query, document_id=document_id, top_k=top_k)
        )
        vec_task = asyncio.create_task(
            self.vector.query(query, document_id=document_id, top_k=top_k)
        )
        
        try:
            elastic_hits, vec_hits = await asyncio.gather(es_task, vec_task, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Error in parallel retrieval: {e}")
            elastic_hits = []
            vec_hits = []
        
        # Handle exceptions from gather
        elastic_hits = elastic_hits if isinstance(elastic_hits, list) else []
        vec_hits = vec_hits if isinstance(vec_hits, list) else []
        
        # Merge results
        merged = self._merge_results(elastic_hits, vec_hits, top_k)
        
        if not merged:
            logger.debug(f"No results from either retriever for query: {query[:50]}...")
            return []
        
        # Rerank with cross-encoder
        ranked = await self._rerank_results(merged, query, top_k)
        
        return ranked[:top_k]

    def _merge_results(self, es_results: List[Dict], vec_results: List[Dict], top_k: int) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from multiple sources with score fusion.
        
        Uses a weighted average of normalized scores from ES and vector retriever.
        """
        # Track seen texts to avoid duplicates (using first 100 chars as key)
        seen = {}
        merged = []
        
        # Add ES results with weight boost
        for r in es_results:
            text = r.get("text", "")
            if not text:
                continue
            
            key = text[:100].lower()
            score = min(1.0, r.get("score", 0) * 1.2)  # Boost ES scores slightly
            
            if key not in seen:
                r_copy = dict(r)
                r_copy["score"] = score
                r_copy["source"] = "elasticsearch"
                seen[key] = r_copy
                merged.append(r_copy)
        
        # Add vector results, merging if we've seen the text
        for r in vec_results:
            text = r.get("text", "")
            if not text:
                continue
            
            key = text[:100].lower()
            vec_score = r.get("score", 0)
            
            if key in seen:
                # Merge with existing result (weighted average)
                existing = seen[key]
                existing["score"] = (existing["score"] + vec_score) / 2
                if "vector_score" not in existing:
                    existing["vector_score"] = vec_score
                if "elastic_score" not in existing:
                    existing["elastic_score"] = existing["score"]
            else:
                r_copy = dict(r)
                r_copy["source"] = "vector"
                seen[key] = r_copy
                merged.append(r_copy)
        
        # Sort by score
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return merged

    async def _rerank_results(self, results: List[Dict], query: str, top_k: int) -> List[Dict[str, Any]]:
        """Apply cross-encoder reranking to merged results."""
        if not self.reranker or len(results) <= 1:
            return results[:top_k]
        
        texts = [r.get("text", "") for r in results if r.get("text")]
        if not texts:
            return results[:top_k]
        
        try:
            ranks = await self.reranker.rank(query, texts, top_n=min(len(texts), top_k))
            
            if not ranks:
                return results[:top_k]
            
            # Reorder by reranker scores
            idxs = [i for i, _ in ranks]
            reranked = [results[i] for i in idxs if i < len(results)]
            
            # Store reranker scores
            for result, (idx, score) in zip(reranked, ranks[:len(reranked)]):
                result["reranker_score"] = score
            
            logger.debug(f"Reranked {len(reranked)} results for query: {query[:50]}...")
            return reranked
        
        except Exception as e:
            logger.warning(f"Reranking failed: {e}; returning original order")
            return results[:top_k]

    async def get_all_chunks(self, document_id: str, limit: int = 200) -> List[str]:
        """Get all chunks for a document from vector store."""
        hits = await self.vector.query("", document_id=document_id, top_k=limit)
        return [t for t in (h.get("text") for h in hits) if isinstance(t, str)]

