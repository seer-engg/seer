"""
Langfuse tracing utilities for LangChain/LangGraph integration.

Supports separate Langfuse projects for:
- Nexus agent tracing
- Workflow/Compiler tracing

Langfuse v3 Multi-Project Pattern:
1. Initialize Langfuse clients with full credentials for each project
2. Create CallbackHandler(public_key=...) to route traces to the correct project
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)


class LangfuseProject(Enum):
    """Langfuse project types for separate tracing."""
    NEXUS = "nexus"
    WORKFLOW = "workflow"


# Singleton instances per project
_langfuse_clients: Dict[LangfuseProject, Any] = {}
_langfuse_handlers: Dict[LangfuseProject, Any] = {}
_handlers_initialized: Dict[LangfuseProject, bool] = {}


def _get_project_credentials(project: LangfuseProject) -> Tuple[Optional[str], Optional[str]]:
    """
    Get credentials for a specific Langfuse project.

    Returns:
        Tuple of (public_key, secret_key) or (None, None) if not configured.
    """
    if project == LangfuseProject.NEXUS:
        if not config.is_langfuse_nexus_configured:
            logger.debug(
                "Langfuse Nexus project not configured. "
                "Set LANGFUSE_NEXUS_PUBLIC_KEY and LANGFUSE_NEXUS_SECRET_KEY."
            )
            return None, None
        return config.langfuse_nexus_public_key, config.langfuse_nexus_secret_key

    # WORKFLOW
    if not config.is_langfuse_workflow_configured:
        logger.debug(
            "Langfuse Workflow project not configured. "
            "Set LANGFUSE_WORKFLOW_PUBLIC_KEY and LANGFUSE_WORKFLOW_SECRET_KEY."
        )
        return None, None
    return config.langfuse_workflow_public_key, config.langfuse_workflow_secret_key


def _create_langfuse_handler(project: LangfuseProject, public_key: str, secret_key: str) -> Optional[Any]:
    """
    Create and cache Langfuse client and callback handler.

    Returns:
        CallbackHandler instance or None on failure.
    """
    try:
        # pylint: disable=import-outside-toplevel  # Reason: lazy loading to avoid dependency if not used
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning("langfuse not installed, skipping Langfuse tracing")
        return None

    try:
        # Initialize Langfuse client with full credentials
        client_kwargs: Dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
        if config.langfuse_host:
            client_kwargs["host"] = config.langfuse_host

        _langfuse_clients[project] = Langfuse(**client_kwargs)

        # Create CallbackHandler with public_key to route to correct project
        # This is required when multiple Langfuse clients exist in the same process
        _langfuse_handlers[project] = CallbackHandler(public_key=public_key)

        logger.info(
            "Langfuse %s tracing initialized (host=%s)",
            project.value,
            config.langfuse_host or "https://cloud.langfuse.com",
        )
        return _langfuse_handlers[project]
    except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: instrumentation should not break agent startup
        logger.warning("Failed to initialize Langfuse %s callback handler: %s", project.value, exc)
        return None


def _get_langfuse_handler(project: LangfuseProject) -> Optional[Any]:
    """
    Get or create Langfuse callback handler for a specific project.

    In Langfuse v3 with multiple projects:
    1. Initialize Langfuse client with full credentials
    2. Create CallbackHandler(public_key=...) to route to correct project

    Args:
        project: Which Langfuse project to use (NEXUS or WORKFLOW)

    Returns:
        CallbackHandler instance or None if not configured/available.
    """
    if _handlers_initialized.get(project, False):
        return _langfuse_handlers.get(project)

    _handlers_initialized[project] = True

    if not config.langfuse_enabled:
        logger.debug("Langfuse tracing disabled via configuration")
        return None

    public_key, secret_key = _get_project_credentials(project)
    if public_key is None or secret_key is None:
        return None

    return _create_langfuse_handler(project, public_key, secret_key)


def get_nexus_langfuse_callbacks() -> List[Any]:
    """
    Get Langfuse callbacks for Nexus agent tracing.

    Returns:
        List containing Langfuse handler if configured, empty list otherwise.
    """
    handler = _get_langfuse_handler(LangfuseProject.NEXUS)
    return [handler] if handler else []


def get_workflow_langfuse_callbacks() -> List[Any]:
    """
    Get Langfuse callbacks for Workflow/Compiler tracing.

    Returns:
        List containing Langfuse handler if configured, empty list otherwise.
    """
    handler = _get_langfuse_handler(LangfuseProject.WORKFLOW)
    return [handler] if handler else []


def merge_nexus_langfuse_callbacks(config_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge Nexus Langfuse callbacks into an existing LangGraph config dictionary.

    Args:
        config_dict: Existing config dictionary (can be None or empty).

    Returns:
        Config dictionary with Langfuse callbacks merged in.
    """
    return _merge_callbacks(config_dict, get_nexus_langfuse_callbacks())


def merge_workflow_langfuse_callbacks(config_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge Workflow Langfuse callbacks into an existing LangGraph config dictionary.

    Args:
        config_dict: Existing config dictionary (can be None or empty).

    Returns:
        Config dictionary with Langfuse callbacks merged in.
    """
    return _merge_callbacks(config_dict, get_workflow_langfuse_callbacks())


def _merge_callbacks(config_dict: Optional[Dict[str, Any]], langfuse_callbacks: List[Any]) -> Dict[str, Any]:
    """
    Internal helper to merge callbacks into config.

    Args:
        config_dict: Existing config dictionary (can be None or empty).
        langfuse_callbacks: List of Langfuse callbacks to merge.

    Returns:
        Config dictionary with callbacks merged in.
    """
    if not langfuse_callbacks:
        return dict(config_dict or {})

    result = dict(config_dict or {})
    existing_callbacks = result.get("callbacks", [])

    # Avoid duplicates if callbacks already include Langfuse handler
    if existing_callbacks:
        for cb in langfuse_callbacks:
            if cb not in existing_callbacks:
                existing_callbacks = list(existing_callbacks) + [cb]
        result["callbacks"] = existing_callbacks
    else:
        result["callbacks"] = langfuse_callbacks

    return result


# Legacy aliases for backwards compatibility
def get_langfuse_callbacks() -> List[Any]:
    """Legacy: Get Langfuse callbacks (defaults to Nexus project)."""
    return get_nexus_langfuse_callbacks()


def merge_langfuse_callbacks(config_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Legacy: Merge Langfuse callbacks (defaults to Nexus project)."""
    return merge_nexus_langfuse_callbacks(config_dict)
