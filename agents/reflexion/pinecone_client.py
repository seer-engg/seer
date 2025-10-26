"""
Pinecone client for storing and retrieving reflection memories.

Uses Pinecone Serverless with OpenAI embeddings for semantic search.
User isolation is achieved through namespaces (user_id as namespace).
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from langchain_openai import OpenAIEmbeddings
from shared.logger import get_logger

logger = get_logger('pinecone_client')

# Pinecone configuration
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "seer-reflexion-memories")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_DUPLICATE_THRESHOLD = float(os.getenv("PINECONE_DUPLICATE_THRESHOLD", "0.9"))
PINECONE_DUPLICATE_TOP_K = int(os.getenv("PINECONE_DUPLICATE_TOP_K", "3"))


def _get_embeddings() -> OpenAIEmbeddings:
    """Get OpenAI embeddings instance for vectorization."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _get_pinecone_client() -> Pinecone:
    """Get Pinecone client with API key from environment."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment")
    return Pinecone(api_key=api_key)


def _get_or_create_index():
    """
    Get or create Pinecone index for reflexion memories.
    
    Uses serverless spec. We'll handle embeddings via OpenAI separately.
    """
    pc = _get_pinecone_client()
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [idx['name'] for idx in existing_indexes]
    
    if PINECONE_INDEX_NAME not in index_names:
        # Create new serverless index
        # Using 1536 dimensions for OpenAI text-embedding-3-small
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
    
    return pc.Index(PINECONE_INDEX_NAME)


def _iso_now() -> str:
    """Return current UTC time in ISO 8601 format with Z suffix."""
    return datetime.utcnow().isoformat() + "Z"


def _normalize_text(text: str) -> str:
    """Normalize text for equality checks (lowercase, collapse whitespace)."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())


def _find_near_duplicate(index, query_vector, namespace: str, threshold: float, top_k: int):
    """Return the best matching vector (>= threshold) or None."""
    try:
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        matches = results.get('matches', []) if isinstance(results, dict) else getattr(results, 'matches', [])
        if not matches:
            return None
        # Keep only matches that meet threshold, pick top by score
        passing = [m for m in matches if (m.get('score') if isinstance(m, dict) else getattr(m, 'score', 0.0)) >= threshold]
        if not passing:
            return None
        # Return the highest score match
        passing.sort(key=lambda m: (m.get('score') if isinstance(m, dict) else getattr(m, 'score', 0.0)), reverse=True)
        return passing[0]
    except Exception as e:
        logger.warning(f"Near-duplicate search failed: {e}")
        return None


def _merge_metadata(existing: Optional[Dict[str, Any]], incoming_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge lifecycle fields and caller metadata for a deduped upsert (no raw text stored)."""
    existing = existing or {}
    merged: Dict[str, Any] = {}
    # Start with existing, then apply caller-provided to allow overrides
    merged.update(existing)
    if incoming_metadata:
        merged.update(incoming_metadata)
    now = _iso_now()
    # Lifecycle fields
    created_at = existing.get('created_at') or now
    times_seen = existing.get('times_seen', 1)
    merged['created_at'] = created_at
    merged['last_updated'] = now
    merged['times_seen'] = times_seen + 1
    return merged


def pinecone_add_memory(
    context: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Add reflection memory to Pinecone.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        user_id: User ID for namespace isolation
        timeout: Request timeout (not used with current SDK, for API compatibility)
    
    Returns:
        Dict with upsert results
    """
    if not user_id:
        user_id = "default"

    try:
        index = _get_or_create_index()
        embeddings = _get_embeddings()

        # Embed the incoming context first
        incoming_vector = embeddings.embed_query(context)

        # Check for near-duplicates within the namespace
        near_dup = _find_near_duplicate(
            index=index,
            query_vector=incoming_vector,
            namespace=user_id,
            threshold=PINECONE_DUPLICATE_THRESHOLD,
            top_k=PINECONE_DUPLICATE_TOP_K,
        )

        # If a near-duplicate exists, merge into it instead of inserting a new vector
        if near_dup is not None:
            existing_id = near_dup.get('id') if isinstance(near_dup, dict) else getattr(near_dup, 'id', None)
            existing_meta = near_dup.get('metadata') if isinstance(near_dup, dict) else getattr(near_dup, 'metadata', {})
            existing_reflection = (existing_meta or {}).get('reflection') or ""

            existing_norm = _normalize_text(existing_reflection)
            context_norm = _normalize_text(context)

            # Prepare merged metadata (no raw text field)
            merged_meta = _merge_metadata(existing_meta, metadata)

            # Case A: Existing record has no reflection -> initialize reflection only, keep vector unchanged
            if not existing_reflection:
                merged_meta['reflection'] = context
                index.update(id=existing_id, set_metadata=merged_meta, namespace=user_id)
                logger.info(f"Deduped memory; initialized reflection on existing id={existing_id} (namespace={user_id}, vector unchanged)")
                return {
                    "success": True,
                    "memory_id": existing_id,
                    "namespace": user_id,
                    "deduped": True
                }

            # Case B: Reflection already contains (or equals) the incoming context -> update counters only
            if existing_norm == context_norm or context_norm in existing_norm:
                index.update(id=existing_id, set_metadata=merged_meta, namespace=user_id)
                logger.info(f"Deduped memory; counters/timestamps updated for existing id={existing_id} (namespace={user_id}, no reflection change)")
                return {
                    "success": True,
                    "memory_id": existing_id,
                    "namespace": user_id,
                    "deduped": True
                }

            # Case C: Append new context to reflection and re-embed merged reflection to keep vector consistent
            merged_reflection = f"{existing_reflection}\n{context}"
            merged_vector = embeddings.embed_query(merged_reflection)
            merged_meta['reflection'] = merged_reflection

            index.upsert(
                vectors=[{
                    "id": existing_id,
                    "values": merged_vector,
                    "metadata": merged_meta
                }],
                namespace=user_id
            )

            logger.info(f"Deduped memory; merged reflection and re-embedded into existing id={existing_id} (namespace={user_id})")

            return {
                "success": True,
                "memory_id": existing_id,
                "namespace": user_id,
                "deduped": True
            }

        # No near-duplicate found; insert as new
        memory_id = str(uuid.uuid4())
        now = _iso_now()
        new_metadata: Dict[str, Any] = {}
        if metadata:
            new_metadata.update(metadata)
        new_metadata.setdefault('created_at', now)
        new_metadata['last_updated'] = now
        new_metadata['times_seen'] = 1

        index.upsert(
            vectors=[{
                "id": memory_id,
                "values": incoming_vector,
                "metadata": new_metadata
            }],
            namespace=user_id
        )

        logger.info(f"Inserted new memory id={memory_id} (namespace={user_id})")

        return {
            "success": True,
            "memory_id": memory_id,
            "namespace": user_id,
            "deduped": False
        }

    except PineconeException as e:
        return {
            "error": f"Pinecone error: {str(e)}",
            "success": False
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "success": False
        }


def pinecone_search_memories(
    query: str, 
    user_id: str, 
    top_k: int = 5,
    timeout: int = 30
) -> List[Dict[str, Any]]:
    """
    Search for relevant reflection memories using semantic search.
    
    Args:
        query: Search query (typically the user's current request)
        user_id: User ID for namespace filtering
        top_k: Number of top results to return
        timeout: Request timeout (not used with current SDK, for API compatibility)
    
    Returns:
        List of matching memories with metadata
    """
    if not user_id:
        user_id = "default"
    
    try:
        index = _get_or_create_index()
        
        # Generate embedding for the query
        embeddings = _get_embeddings()
        query_vector = embeddings.embed_query(query)
        logger.info(f"Searching for {query} in Pinecone")
        
        # Query Pinecone with the embedded vector
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=user_id,
            include_metadata=True
        )
        logger.info(f"Pinecone search results: {results}")
        # Extract and format results        
        return results.get('matches', [])
    
    except PineconeException as e:
        print(f"Pinecone search error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected search error: {e}")
        return []


def pinecone_delete_user_memories(user_id: str) -> Dict[str, Any]:
    """
    Delete all memories for a specific user (namespace).
    
    Args:
        user_id: User ID whose memories should be deleted
    
    Returns:
        Dict with deletion status
    """
    try:
        index = _get_or_create_index()
        
        # Delete all vectors in the namespace
        index.delete(delete_all=True, namespace=user_id)
        
        return {
            "success": True,
            "message": f"Deleted all memories for user {user_id}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


