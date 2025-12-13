import os
import asyncio
from typing import Optional, List, Any
from shared.tools import ComposioMCPClient
from shared.config import config
from agents.eval_agent.models import TestExecutionState
from tool_hub import ToolHub
from tool_hub.models import Tool, ToolFunction
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

# Cache the hub instance to avoid reloading index on every call
_CACHED_HUB: Optional[ToolHub] = None
TOOL_HUB_INDEX_DIR = config.tool_hub_index_dir

def _convert_tools_for_ingestion(tools: List[Any]) -> List[Tool]:
    """Converts LangChain tools to ToolHub schema."""
    normalized = []
    for t in tools:
        try:
            # Handle LangChain BaseTool
            name = getattr(t, "name", None)
            desc = getattr(t, "description", "")
            args = getattr(t, "args", {})
            
            if name:
                normalized.append(Tool(
                    function=ToolFunction(
                        name=name,
                        description=desc,
                        parameters=args
                    ),
                    executable=t
                ))
        except Exception as e:
            print(f"Skipping tool conversion for {t}: {e}")
    return normalized

async def get_tool_hub() -> ToolHub:
    """
    Returns a fully initialized and hydrated ToolHub.
    """
    global _CACHED_HUB
    if _CACHED_HUB:
        return _CACHED_HUB

    # 1. Initialize Hub
    openai_key = config.openai_api_key
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    hub = ToolHub(openai_api_key=openai_key)


    # TODO: replace with vector index search
    tool_service = ComposioMCPClient(["GITHUB", "ASANA"], config.composio_user_id)
    all_tools = await tool_service.get_tools()

    # 3. Load or Ingest
    if os.path.exists(TOOL_HUB_INDEX_DIR) and os.path.exists(os.path.join(TOOL_HUB_INDEX_DIR, "metadata.json")):
        # Load metadata/index from disk (Fast)
        try:
            hub.load(TOOL_HUB_INDEX_DIR)
        except Exception as e:
            print(f"Failed to load ToolHub index: {e}. Re-ingesting.")
            ingest_tools = _convert_tools_for_ingestion(all_tools)
            # Run potentially blocking ingestion in a background thread to avoid
            # blocking the LangGraph event loop (see blockbuster.BlockingError).
            await asyncio.to_thread(hub.ingest, ingest_tools)
            await asyncio.to_thread(hub.save, TOOL_HUB_INDEX_DIR)
    else:
        # First run: Ingest everything (Slow, but one-time)
        print("Initializing ToolHub index (first run)...")
        ingest_tools = _convert_tools_for_ingestion(all_tools)
        # Same as above: move blocking ingestion + disk I/O off the main loop.
        await asyncio.to_thread(hub.ingest, ingest_tools)
        await asyncio.to_thread(hub.save, TOOL_HUB_INDEX_DIR)

    # 4. Bind Executables (Critical Step)
    # Match the live functions to the loaded metadata
    hub.bind_executables(all_tools)
    
    _CACHED_HUB = hub
    return hub



@wrap_tool_call
async def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return await handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )