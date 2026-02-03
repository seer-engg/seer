"""Embedding service for generating vector embeddings using OpenAI."""
from __future__ import annotations

from typing import List

from langchain_openai import OpenAIEmbeddings

from seer.config import config
from seer.logger import get_logger

logger = get_logger("services.knowledge.embedding")


class EmbeddingService:
    """Service for generating text embeddings using OpenAI API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
    ):
        """Initialize embedding service with configuration.

        Args:
            model: OpenAI embedding model name (default from config)
            api_key: OpenAI API key (default from config)
            dimensions: Embedding dimensions (default from config)
            batch_size: Batch size for embedding requests (default from config)
        """
        self.model = model or config.embedding_model
        self.dimensions = dimensions or config.embedding_dims
        self.batch_size = batch_size or config.embedding_batch_size
        self._api_key = api_key or config.openai_api_key

        if not self._api_key:
            raise ValueError("OpenAI API key is required for embedding service")

        self._embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self._api_key,
            dimensions=self.dimensions,
        )

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        embeddings = await self._embeddings.aembed_documents([text])
        return embeddings[0]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embeddings.aembed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            logger.debug(
                "Embedded batch",
                extra={"batch_start": i, "batch_size": len(batch), "total": len(texts)},
            )

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query (may use different model internally).

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        return await self._embeddings.aembed_query(query)


# Singleton instance
_EMBEDDING_SERVICE: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create singleton embedding service."""
    global _EMBEDDING_SERVICE  # pylint: disable=global-statement  # Reason: Singleton pattern for service instance
    if _EMBEDDING_SERVICE is None:
        _EMBEDDING_SERVICE = EmbeddingService()
    return _EMBEDDING_SERVICE


__all__ = ["EmbeddingService", "get_embedding_service"]
