"""Shared LocalToolHub singleton instance."""
from typing import Optional
from shared.config import config
from shared.logger import get_logger
from .local_core import LocalToolHub

logger = get_logger("shared.tool_hub.singleton")

# Singleton container to avoid using 'global' in functions
_TOOLHUB_CONTAINER: dict[str, Optional[LocalToolHub]] = {"instance": None}

def get_toolhub_instance() -> Optional[LocalToolHub]:
    """Get or create the shared LocalToolHub singleton instance."""
    if _TOOLHUB_CONTAINER["instance"] is None:
        try:
            if not config.openai_api_key:
                logger.debug("OpenAI API key not configured, tool search will use fallback")
                return None

            _TOOLHUB_CONTAINER["instance"] = LocalToolHub(
                openai_api_key=config.openai_api_key,
                persist_directory=config.tool_index_path,
                llm_model=config.default_llm_model,
                embedding_model=config.embedding_model,
                embedding_dimensions=config.embedding_dims,
            )
            logger.info("✅ Shared LocalToolHub singleton initialized")
        except Exception as e:
            logger.warning("LocalToolHub initialization failed: %s", e)
            return None

    return _TOOLHUB_CONTAINER["instance"]

def set_toolhub_instance(instance: LocalToolHub) -> None:
    """Set the shared instance (for testing or pre-initialization)."""
    _TOOLHUB_CONTAINER["instance"] = instance
