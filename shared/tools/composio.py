"""
Central client for managing MCP tool access via Composio.

Reads a central mcp.json configuration file to determine which services
to expose, then uses Composio (with the LangChain provider) to return
LangChain-compatible tools. Caches clients and tool lists per service set.
"""
import asyncio
import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
from langchain_core.tools import BaseTool
from shared.logger import get_logger
from composio import Composio
from composio_langchain import LangchainProvider

logger = get_logger("shared.tools.composio")


class ComposioMCPClient:
    """
    Lightweight shim that matches the `.get_tools()` interface expected by callers,
    backed by Composio's LangChain provider.
    """
    def __init__(self, service_names: List[str], user_id: str):
        self._service_names = [str(s or "").strip().lower() for s in service_names or []]
        self._user_id = user_id
        self._composio = Composio(provider=LangchainProvider())

    @staticmethod
    def _to_toolkits(service_names: List[str]) -> List[str]:
        toolkits: List[str] = []
        for name in service_names:
            canonical = str(name).replace("-", "_").upper()
            toolkits.append(canonical)
        return toolkits

    async def get_tools(self) -> List[BaseTool]:
        toolkits = self._to_toolkits(self._service_names)
        tools = await asyncio.to_thread(
            self._composio.tools.get,
            user_id=self._user_id or "default",
            toolkits=toolkits,
            limit=2000
        )
        return tools
    
    def get_client(self) -> Composio:
        return self._composio
