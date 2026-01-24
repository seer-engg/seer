# pylint: disable=duplicate-code  # Reason: Shared singleton initialization logic is mirrored in tool_hub.singleton
"""
Tool index management utilities.

Handles generation and loading of tool vector index during startup.
"""
import threading
from typing import Any, Dict, List, Optional

from seer.config import config
from seer.logger import get_logger
from seer.tool_hub.local_core import LocalToolHub
from seer.tool_hub.models import Tool, ToolFunction
from seer.tools.registry import get_tools_by_integration

logger = get_logger("shared.tool_hub.index_manager")


def _group_tools_by_integration(all_tools_meta: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    tools_by_integration: Dict[str, List[Dict[str, Any]]] = {}
    for tool_meta in all_tools_meta:
        integration_type = tool_meta.get("integration_type", "unknown")
        tools_by_integration.setdefault(integration_type, []).append(tool_meta)
    return tools_by_integration


def _build_tool_objects(
    tools_by_integration: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Tool]]:
    tools_by_integration_objects: Dict[str, List[Tool]] = {}
    for integration_type, tool_meta_list in tools_by_integration.items():
        tool_objects = []
        for tool_meta in tool_meta_list:
            tool_function = ToolFunction(
                name=tool_meta.get("name", ""),
                description=tool_meta.get("description", ""),
                parameters=tool_meta.get("parameters", {}),
            )
            tool_objects.append(Tool(function=tool_function))
        tools_by_integration_objects[integration_type] = tool_objects
    return tools_by_integration_objects


def _ingest_tools(toolhub: LocalToolHub, tools_by_integration_objects: Dict[str, List[Tool]]) -> None:
    for integration_type, tools in tools_by_integration_objects.items():
        logger.info("Ingesting %s tools for integration: %s", len(tools), integration_type)
        try:
            threading.Thread(target=toolhub.ingest, args=(tools, integration_type)).start()
        except Exception as exc:  # pylint: disable=broad-exception-caught  # ingestion should not crash startup
            logger.error("Failed to ingest tools for %s: %s", integration_type, exc)


async def generate_tool_index(
    toolhub: LocalToolHub,
    force_regenerate: bool = False,
) -> bool:
    """
    Generate tool index from all registered tools.

    Args:
        toolhub: LocalToolHub instance to use for storage.
        force_regenerate: If True, regenerate even if index exists.

    Returns:
        True if index was generated successfully, False otherwise.
    """
    try:
        # Check if index already exists
        if not force_regenerate and toolhub.index_exists():
            logger.info("Tool index already exists, skipping generation.")
            stats = toolhub.get_index_stats()
            logger.info("Index stats: %s", stats)
            return True

        logger.info("Starting tool index generation...")

        all_tools_meta = get_tools_by_integration()

        if not all_tools_meta:
            logger.warning("No tools found in registry. Cannot generate index.")
            return False

        tools_by_integration = _group_tools_by_integration(all_tools_meta)
        logger.info(
            "Found %s tools across %s integrations",
            len(all_tools_meta),
            len(tools_by_integration),
        )

        tools_by_integration_objects = _build_tool_objects(tools_by_integration)
        _ingest_tools(toolhub, tools_by_integration_objects)
        return True

    except Exception as exc:  # pylint: disable=broad-exception-caught  # protect startup path
        logger.exception("Error generating tool index")
        logger.debug("Tool index generation failure details: %s", exc)
        return False


async def ensure_tool_index_exists(
    toolhub: Optional[LocalToolHub] = None,
    auto_generate: bool = True
) -> Optional[LocalToolHub]:
    """
    Ensure tool index exists, generating it if necessary.

    Args:
        toolhub: Optional LocalToolHub instance. If None, creates a new one.
        auto_generate: If True, automatically generate index if missing.

    Returns:
        LocalToolHub instance, or None if initialization failed.
    """
    try:
        # Create toolhub if not provided
        if toolhub is None:
            if not config.openai_api_key:
                logger.warning("OpenAI API key not configured. Cannot initialize tool index.")
                return None

            toolhub = LocalToolHub(
                openai_api_key=config.openai_api_key,
                persist_directory=config.tool_index_path,
                llm_model=config.default_llm_model,
                embedding_model=config.embedding_model,
                embedding_dimensions=config.embedding_dims,
            )

        # Check if index exists
        if toolhub.index_exists():
            logger.info("Tool index found and loaded.")
            stats = toolhub.get_index_stats()
            logger.info("Index stats: %s", stats)
            return toolhub

        # Generate index if auto_generate is enabled
        if auto_generate:
            logger.info("Tool index not found. Generating...")
            success = await generate_tool_index(toolhub, force_regenerate=False)
            if success:
                return toolhub
            logger.error("Failed to generate tool index.")
            return None
        logger.warning("Tool index not found and auto_generate is disabled.")
        return None

    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: initialization failures should not crash startup
        logger.exception("Error ensuring tool index exists: %s", e)
        return None
