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
        # Generate unique ID for this reflection
        memory_id = f"reflection_{uuid.uuid4().hex}_{int(datetime.now().timestamp())}"
        
        # Generate embedding for the reflection text
        embeddings = _get_embeddings()
        embedding_vector = embeddings.embed_query(context)
        
        # Prepare record for upsert
        record = {
            "id": memory_id,
            "values": embedding_vector,
            "metadata": metadata
        }
        
        # Upsert to Pinecone with user_id as namespace
        index.upsert(
            vectors=[record],
            namespace=user_id
        )
        
        return {
            "success": True,
            "memory_id": memory_id,
            "namespace": user_id
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


