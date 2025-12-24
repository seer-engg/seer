"""Shared LocalToolHub singleton instance."""
from typing import Optional
from .local_core import LocalToolHub
from shared.config import config
from shared.logger import get_logger

logger = get_logger("shared.tool_hub.singleton")

# Global singleton instance
_TOOLHUB_INSTANCE: Optional[LocalToolHub] = None

def get_toolhub_instance() -> Optional[LocalToolHub]:
    """Get or create the shared LocalToolHub singleton instance."""
    global _TOOLHUB_INSTANCE
    if _TOOLHUB_INSTANCE is None:
        try:
            if not config.openai_api_key:
                logger.debug("OpenAI API key not configured, tool search will use fallback")
                return None
            
            _TOOLHUB_INSTANCE = LocalToolHub(
                openai_api_key=config.openai_api_key,
                persist_directory=config.tool_index_path,
                llm_model=config.default_llm_model,
                embedding_model=config.embedding_model,
                embedding_dimensions=config.embedding_dims,
            )
            logger.info("✅ Shared LocalToolHub singleton initialized")
        except Exception as e:
            logger.warning(f"LocalToolHub initialization failed: {e}")
            return None
    
    return _TOOLHUB_INSTANCE

def set_toolhub_instance(instance: LocalToolHub) -> None:
    """Set the shared instance (for testing or pre-initialization)."""
    global _TOOLHUB_INSTANCE
    _TOOLHUB_INSTANCE = instance

