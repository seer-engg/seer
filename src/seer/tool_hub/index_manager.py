"""
Tool index management utilities.

Handles generation and loading of tool vector index during startup.
"""
import threading
from typing import Any, Dict, List, Optional

from seer.config import config
from seer.logger import get_logger
from seer.tool_hub.local_core import LocalToolHub
from seer.tool_hub.models import Tool
from seer.tools.registry import get_tools_by_integration

logger = get_logger("shared.tool_hub.index_manager")


async def generate_tool_index(
    toolhub: LocalToolHub,
    force_regenerate: bool = False
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

        # Get all tools from registry
        all_tools_meta = get_tools_by_integration()

        if not all_tools_meta:
            logger.warning("No tools found in registry. Cannot generate index.")
            return False

        # Group tools by integration type
        tools_by_integration: Dict[str, List[Dict[str, Any]]] = {}

        for tool_meta in all_tools_meta:
            integration_type = tool_meta.get("integration_type", "unknown")
            if integration_type not in tools_by_integration:
                tools_by_integration[integration_type] = []
            tools_by_integration[integration_type].append(tool_meta)

        logger.info("Found %s tools across {len(tools_by_integration)} integrations", len(all_tools_meta))

        # Convert tool metadata to Tool objects
        # We need to convert the metadata dicts to Tool objects
        from seer.tool_hub.models import ToolFunction

        tools_by_integration_objects: Dict[str, List[Tool]] = {}

        for integration_type, tool_meta_list in tools_by_integration.items():
            tool_objects = []
            for tool_meta in tool_meta_list:
                # Convert metadata dict to Tool object
                tool_function = ToolFunction(
                    name=tool_meta.get("name", ""),
                    description=tool_meta.get("description", ""),
                    parameters=tool_meta.get("parameters", {}),
                )
                tool_obj = Tool(function=tool_function)
                tool_objects.append(tool_obj)
            tools_by_integration_objects[integration_type] = tool_objects

        # Ingest tools for each integration
        for integration_type, tools in tools_by_integration_objects.items():
            logger.info("Ingesting %s tools for integration: {integration_type}", len(tools))
            try:
                threading.Thread(target=toolhub.ingest, args=(tools, integration_type)).start()
            except Exception:
                logger.error("Failed to ingest tools for %s: {e}", integration_type)
                continue

    except Exception:
        logger.exception("Error generating tool index")
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

    except Exception as e:
        logger.exception("Error ensuring tool index exists: %s", e)
        return None
