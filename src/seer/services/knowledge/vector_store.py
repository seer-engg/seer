"""Vector store service for pgvector operations."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tortoise import Tortoise

from seer.logger import get_logger

logger = get_logger("services.knowledge.vector_store")


class PgVectorStore:
    """Service for storing and querying vector embeddings using pgvector."""

    async def similarity_search(
        self,
        kb_id: int,
        embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Perform similarity search on knowledge base chunks.

        Args:
            kb_id: Knowledge base internal ID
            embedding: Query embedding vector
            top_k: Maximum number of results to return
            min_score: Minimum similarity score threshold (0-1)

        Returns:
            List of matching chunks with scores
        """
        # Validate numeric params at the DB boundary
        if not isinstance(min_score, (int, float)):
            min_score = 0.7
        if not isinstance(top_k, int):
            top_k = 5

        conn = Tortoise.get_connection("default")

        # Convert embedding to PostgreSQL array format
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        # Using cosine distance: 1 - (a <=> b) gives similarity score
        query = """
            SELECT
                c.id as chunk_id,
                c.content,
                c.metadata as chunk_metadata,
                c.chunk_index,
                d.id as doc_id,
                d.name as doc_name,
                1 - (c.embedding <=> $1::vector) as score
            FROM knowledge_chunks c
            JOIN knowledge_documents d ON c.document_id = d.id
            WHERE c.knowledge_base_id = $2
                AND c.embedding IS NOT NULL
                AND 1 - (c.embedding <=> $1::vector) >= $3
            ORDER BY c.embedding <=> $1::vector
            LIMIT $4
        """

        results = await conn.execute_query_dict(query, [embedding_str, kb_id, min_score, top_k])

        return [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "doc_name": row["doc_name"],
                "content": row["content"],
                "score": float(row["score"]),
                "metadata": row["chunk_metadata"],
                "chunk_index": row["chunk_index"],
            }
            for row in results
        ]

    async def insert_chunks(
        self,
        document_id: int,
        kb_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
        *,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Insert chunks with embeddings into the vector store.

        Args:
            document_id: Document internal ID
            kb_id: Knowledge base internal ID
            chunks: List of text chunks
            embeddings: List of embedding vectors (must match chunks length)
            metadata: Optional list of metadata dicts for each chunk

        Returns:
            Number of chunks inserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have same length")

        if not chunks:
            return 0

        conn = Tortoise.get_connection("default")
        metadata = metadata or [None] * len(chunks)

        # Build batch insert query
        values_parts = []
        params: List[Any] = []
        param_idx = 1

        for i, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, metadata)):
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            values_parts.append(
                f"(${param_idx}, ${param_idx + 1}, ${param_idx + 2}, ${param_idx + 3}::vector, "
                f"${param_idx + 4}::jsonb, ${param_idx + 5})"
            )
            params.extend([document_id, kb_id, i, embedding_str, meta, chunk])
            param_idx += 6

        query = f"""
            INSERT INTO knowledge_chunks
                (document_id, knowledge_base_id, chunk_index, embedding, metadata, content)
            VALUES {", ".join(values_parts)}
        """

        await conn.execute_query(query, params)
        logger.info(
            "Inserted chunks",
            extra={"document_id": document_id, "kb_id": kb_id, "chunk_count": len(chunks)},
        )

        return len(chunks)

    async def delete_document_chunks(self, document_id: int) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Document internal ID

        Returns:
            Number of chunks deleted
        """
        conn = Tortoise.get_connection("default")
        result = await conn.execute_query(
            "DELETE FROM knowledge_chunks WHERE document_id = $1 RETURNING id",
            [document_id],
        )
        deleted_count = len(result[1]) if result[1] else 0
        logger.info("Deleted chunks", extra={"document_id": document_id, "deleted_count": deleted_count})
        return deleted_count

    async def delete_kb_chunks(self, kb_id: int) -> int:
        """Delete all chunks for a knowledge base.

        Args:
            kb_id: Knowledge base internal ID

        Returns:
            Number of chunks deleted
        """
        conn = Tortoise.get_connection("default")
        result = await conn.execute_query(
            "DELETE FROM knowledge_chunks WHERE knowledge_base_id = $1 RETURNING id",
            [kb_id],
        )
        deleted_count = len(result[1]) if result[1] else 0
        logger.info("Deleted KB chunks", extra={"kb_id": kb_id, "deleted_count": deleted_count})
        return deleted_count

    async def get_chunk_count(self, kb_id: int) -> int:
        """Get total chunk count for a knowledge base.

        Args:
            kb_id: Knowledge base internal ID

        Returns:
            Total number of chunks
        """
        conn = Tortoise.get_connection("default")
        result = await conn.execute_query_dict(
            "SELECT COUNT(*) as count FROM knowledge_chunks WHERE knowledge_base_id = $1",
            [kb_id],
        )
        return result[0]["count"] if result else 0


# Singleton instance
_VECTOR_STORE: PgVectorStore | None = None


def get_vector_store() -> PgVectorStore:
    """Get or create singleton vector store."""
    global _VECTOR_STORE  # pylint: disable=global-statement  # Reason: Singleton pattern for service instance
    if _VECTOR_STORE is None:
        _VECTOR_STORE = PgVectorStore()
    return _VECTOR_STORE


__all__ = ["PgVectorStore", "get_vector_store"]
