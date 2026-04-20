import asyncio
import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning("Failed to load CrossEncoder: %s", e)
                self._model = None

    async def rank(self, query: str, chunks: List[str], top_n: int = 5) -> List[Tuple[int, float]]:
        """Return list of (index, score) sorted desc by score for top_n"""
        def _sync_rank():
            self._ensure()
            if not self._model:
                return []
            pairs = [[query, c] for c in chunks]
            try:
                scores = self._model.predict(pairs)
            except Exception as e:
                logger.exception("Reranker predict failed: %s", e)
                return []
            indexed = list(enumerate(scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            return indexed[:top_n]

        return await asyncio.to_thread(_sync_rank)
