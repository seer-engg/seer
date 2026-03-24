"""In-memory semantic search over tools and triggers using OpenAI embeddings."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np

from seer.logger import get_logger

logger = get_logger("tools.semantic_index")

_INSTANCE: Optional[ToolSemanticIndex] = None
_LOCK = asyncio.Lock()


class ToolSemanticIndex:
    """Embeds all tools + triggers at first use, then answers queries via cosine similarity."""

    def __init__(self) -> None:
        self._embeddings: Optional[np.ndarray] = None  # (N, dims)
        self._items: List[Dict[str, Any]] = []
        self.is_initialized = False

    async def build_index(self) -> None:
        """Embed all tools + triggers. Called once (~500ms, ~$0.001)."""
        from seer.tools.registry import get_tools_by_integration  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import
        from seer.core.registry.trigger_registry import trigger_registry  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import
        from seer.services.knowledge.embedding_service import EmbeddingService  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import

        items: List[Dict[str, Any]] = []
        texts: List[str] = []

        # Tools
        for t in get_tools_by_integration():
            name = t.get("name", "")
            desc = t.get("description", "")
            items.append({
                "name": name,
                "description": desc,
                "integration_type": t.get("integration_type", ""),
                "item_type": "tool",
            })
            texts.append(f"{name}: {desc}")

        # Triggers
        for t in trigger_registry.all():
            items.append({
                "name": t.key,
                "description": t.description or t.title,
                "integration_type": t.provider,
                "item_type": "trigger",
            })
            texts.append(f"{t.key}: {t.title}. {t.description or ''}")

        if not texts:
            self._items = []
            self._embeddings = np.empty((0, 0))
            self.is_initialized = True
            return

        try:
            svc = EmbeddingService()
            vectors = await svc.embed_texts(texts)
            self._embeddings = np.array(vectors, dtype=np.float32)
            self._items = items
            self.is_initialized = True
            logger.info("Semantic index built: %d items", len(items))
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: index build is non-critical
            logger.exception("Failed to build semantic index")
            self.is_initialized = False

    async def search(self, query: str, top_k: int = 10, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Embed query, cosine sim against index, return top-k matches.

        Args:
            query: Natural language search query
            top_k: Max results
            item_type: Filter to "tool" or "trigger" only

        Returns:
            List of dicts with name, description, integration_type, item_type, score
        """
        if not self.is_initialized or self._embeddings is None or len(self._items) == 0:
            return []

        from seer.services.knowledge.embedding_service import EmbeddingService  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import

        try:
            svc = EmbeddingService()
            query_vec = np.array(await svc.embed_text(query), dtype=np.float32)

            # Cosine similarity
            norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec)
            norms = np.where(norms == 0, 1, norms)
            scores = self._embeddings @ query_vec / norms

            # Filter by item_type if specified
            if item_type:
                mask = np.array([it["item_type"] == item_type for it in self._items])
                scores = np.where(mask, scores, -1)

            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for i in top_indices:
                if scores[i] <= 0:
                    break
                results.append({
                    **self._items[i],
                    "score": float(scores[i]),
                })
            return results
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: search failure should not crash the agent
            logger.exception("Semantic search failed")
            return []


async def get_semantic_index() -> ToolSemanticIndex:
    """Get or create the singleton semantic index."""
    global _INSTANCE  # pylint: disable=global-statement  # Reason: singleton pattern for in-memory index
    if _INSTANCE is not None and _INSTANCE.is_initialized:
        return _INSTANCE

    async with _LOCK:
        if _INSTANCE is not None and _INSTANCE.is_initialized:
            return _INSTANCE
        _INSTANCE = ToolSemanticIndex()
        await _INSTANCE.build_index()
        return _INSTANCE
