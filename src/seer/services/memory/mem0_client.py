"""
Singleton Mem0 client for memory operations.

Provides a configured Mem0 Memory instance based on application settings.
Uses lazy initialization to avoid startup overhead when memory is disabled.
"""

import os
from typing import Optional
from urllib.parse import urlparse
import threading

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton
_MEM0_CLIENT: Optional["Memory"] = None  # type: ignore[name-defined]
_client_lock = threading.Lock()


def _disable_mem0_telemetry_store(client: "Memory") -> None:  # type: ignore[name-defined]
    """
    Disable Mem0's auxiliary telemetry vector store.

    Mem0 0.1.115 creates an internal `mem0migrations` pgvector collection for
    anonymous telemetry. In long-lived dev databases that table can be left at a
    stale embedding dimension, which then poisons the telemetry connection with
    repeated transaction-aborted errors. The product memory path uses
    `client.vector_store`, so disabling only the telemetry store is safe.
    """
    if hasattr(client, "_telemetry_vector_store"):
        setattr(client, "_telemetry_vector_store", None)


def get_mem0_client() -> Optional["Memory"]:  # type: ignore[name-defined]
    """
    Get or create the singleton Mem0 client.

    Returns None if memory is disabled via config.
    Uses double-checked locking for thread-safe lazy initialization.

    Returns:
        Configured Mem0 Memory instance, or None if disabled
    """
    global _MEM0_CLIENT  # pylint: disable=global-statement  # Reason: Singleton pattern requires module-level state

    if not config.memory_enabled:
        return None

    if _MEM0_CLIENT is not None:
        return _MEM0_CLIENT

    with _client_lock:
        # Double-check after acquiring lock
        if _MEM0_CLIENT is not None:
            return _MEM0_CLIENT

        try:
            # Set environment variables that LiteLLM expects
            # LiteLLM reads API keys from os.environ, not from Pydantic config
            if config.openrouter_api_key:
                os.environ["OPENROUTER_API_KEY"] = config.openrouter_api_key
            if config.openai_api_key:
                os.environ["OPENAI_API_KEY"] = config.openai_api_key

            from mem0 import Memory  # pylint: disable=import-outside-toplevel  # Reason: Lazy import to avoid startup cost when disabled

            mem0_config = _build_mem0_config()
            logger.info(
                "Initializing Mem0 with llm_provider=%s, embedder=%s, vector_store=%s",
                mem0_config["llm"]["provider"],
                mem0_config["embedder"]["provider"],
                mem0_config["vector_store"]["provider"],
            )
            _MEM0_CLIENT = Memory.from_config(mem0_config)
            _disable_mem0_telemetry_store(_MEM0_CLIENT)
            logger.info(
                "Mem0 client initialized successfully with collection=%s",
                config.mem0_collection_name,
            )
            return _MEM0_CLIENT

        except ImportError as e:
            logger.error("Failed to import mem0: %s. Memory features disabled.", e)
            return None
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Mem0 init can fail for many reasons (network, config, etc.)
            logger.error("Failed to initialize Mem0 client: %s. Memory features disabled.", e)
            return None


def _parse_database_url(url: str) -> dict:
    """
    Parse PostgreSQL URL into individual connection parameters.

    Mem0's pgvector config requires individual fields (host, port, user, password, dbname)
    rather than a connection string.

    Args:
        url: PostgreSQL connection URL (e.g., postgresql://user:pass@host:port/dbname)

    Returns:
        Dict with host, port, user, password, dbname
    """
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/") or "postgres",
    }


def _build_mem0_config() -> dict:
    """
    Build Mem0 configuration dict from app settings.

    Supports multiple vector store backends:
    - pgvector (recommended): Reuses existing PostgreSQL
    - qdrant: Best for very large deployments
    - chroma: Good for local development

    Supports multiple LLM/embedder providers:
    - openai: Direct OpenAI API
    - openrouter: OpenRouter (OpenAI-compatible)
    - huggingface: Free HuggingFace embeddings
    - ollama: Local Ollama models
    """
    vector_store_config = {
        "provider": config.mem0_vector_store,
        "config": {
            "collection_name": config.mem0_collection_name,
            "embedding_model_dims": _get_embedding_dims(),
        }
    }

    # Add provider-specific settings for vector store
    if config.mem0_vector_store == "qdrant":
        vector_store_config["config"]["host"] = config.mem0_qdrant_host
        vector_store_config["config"]["port"] = config.mem0_qdrant_port
    elif config.mem0_vector_store == "pgvector":
        # Reuse existing database URL for pgvector
        # Mem0 requires individual fields (host, port, user, password, dbname)
        if config.database_url:
            db_params = _parse_database_url(config.database_url)
            vector_store_config["config"].update(db_params)

    return {
        "vector_store": vector_store_config,
        "llm": _build_llm_config(),
        "embedder": _build_embedder_config(),
    }


def _get_embedding_dims() -> int:
    """Get embedding dimensions based on the embedder model."""
    # HuggingFace sentence-transformers models typically use 384 dims
    if config.mem0_embedder_provider == "huggingface":
        if "all-MiniLM" in config.mem0_embedder_model:
            return 384
        if "all-mpnet" in config.mem0_embedder_model:
            return 768
        # Default for unknown HuggingFace models
        return 384
    return config.mem0_embedding_dims


def _build_llm_config() -> dict:
    """
    Build LLM configuration for Mem0 based on provider.

    Supported providers:
    - litellm: Universal interface supporting 100+ providers including OpenRouter
    - openrouter: Uses litellm with openrouter/ prefix
    - ollama: Local models
    - openai: Direct OpenAI API
    """
    provider = str(config.mem0_llm_provider).lower()

    if provider in ("openrouter", "litellm"):
        # Use litellm provider which supports OpenRouter via "openrouter/<model>" format
        # See: https://docs.litellm.ai/docs/providers/openrouter
        model_name = str(config.memory_extraction_model)

        # If using OpenRouter but model doesn't have prefix, add it
        if provider == "openrouter" and not model_name.startswith("openrouter/"):
            model_name = f"openrouter/{model_name}"

        return {
            "provider": "litellm",
            "config": {
                "model": model_name,
                "temperature": 0.1,
                "max_tokens": 2000,
            }
        }
    if provider == "ollama":
        return {
            "provider": "ollama",
            "config": {
                "model": config.memory_extraction_model,
                "ollama_base_url": config.mem0_llm_base_url or "http://localhost:11434",
            }
        }
    # Default: OpenAI
    llm_config: dict = {
        "provider": "openai",
        "config": {
            "model": config.memory_extraction_model,
        }
    }
    if config.openai_api_key:
        llm_config["config"]["api_key"] = config.openai_api_key
    if config.mem0_llm_base_url:
        llm_config["config"]["openai_base_url"] = config.mem0_llm_base_url
    return llm_config


def _build_embedder_config() -> dict:
    """Build embedder configuration for Mem0 based on provider."""
    provider = str(config.mem0_embedder_provider).lower()

    if provider == "huggingface":
        # HuggingFace embeddings (free, no API key needed)
        return {
            "provider": "huggingface",
            "config": {
                "model": config.mem0_embedder_model,
            }
        }
    if provider == "ollama":
        return {
            "provider": "ollama",
            "config": {
                "model": config.mem0_embedder_model,
                "ollama_base_url": config.mem0_llm_base_url or "http://localhost:11434",
            }
        }
    # Default: OpenAI
    embedder_config: dict = {
        "provider": "openai",
        "config": {
            "model": config.mem0_embedder_model,
        }
    }
    if config.openai_api_key:
        embedder_config["config"]["api_key"] = config.openai_api_key
    return embedder_config


def reset_mem0_client() -> None:
    """
    Reset the singleton client (for testing or reconfiguration).
    """
    global _MEM0_CLIENT  # pylint: disable=global-statement  # Reason: Singleton pattern requires module-level state
    with _client_lock:
        _MEM0_CLIENT = None
        logger.debug("Mem0 client reset")
