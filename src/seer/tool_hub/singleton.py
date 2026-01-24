"""Shared LocalToolHub singleton instance."""
from typing import Optional

from seer.config import config
from seer.logger import get_logger

from .local_core import LocalToolHub

logger = get_logger("shared.tool_hub.singleton")


class _ToolHubState:
    instance: Optional[LocalToolHub] = None


def _create_toolhub_instance() -> Optional[LocalToolHub]:
    if not config.openai_api_key:
        logger.debug("OpenAI API key not configured, tool search will use fallback")
        return None

    return LocalToolHub(
        openai_api_key=config.openai_api_key,
        persist_directory=config.tool_index_path,
        llm_model=config.default_llm_model,
        embedding_model=config.embedding_model,
        embedding_dimensions=config.embedding_dims,
    )


def get_toolhub_instance() -> Optional[LocalToolHub]:
    """Get or create the shared LocalToolHub singleton instance."""
    if _ToolHubState.instance is None:
        try:
            _ToolHubState.instance = _create_toolhub_instance()
            if _ToolHubState.instance:
                logger.info("✅ Shared LocalToolHub singleton initialized")
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Initialization may fail on missing deps
            logger.warning("LocalToolHub initialization failed: %s", exc)
            return None

    return _ToolHubState.instance


def set_toolhub_instance(instance: LocalToolHub) -> None:
    """Set the shared instance (for testing or pre-initialization)."""
    _ToolHubState.instance = instance
