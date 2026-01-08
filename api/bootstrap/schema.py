"""
Bootstrap endpoint schemas.

This module defines the response types for the /api/bootstrap endpoint,
which consolidates multiple API calls into a single request.
"""
from typing import List, Any, Dict, Optional
from pydantic import BaseModel


class BootstrapResponse(BaseModel):
    """
    Consolidated bootstrap data from multiple endpoints.

    Includes:
    - tools: List of available tools with metadata
    - models: List of available LLM models
    - tools_status: Connection status for each tool
    - connections: User's integration connections
    - node_types: Available workflow node types
    - workflows: User's workflows
    """
    # From /api/tools
    tools: List[Dict[str, Any]] = []

    # From /api/models
    models: List[Dict[str, Any]] = []

    # From /api/integrations/tools/status
    tools_status: List[Dict[str, Any]] = []

    # From /api/integrations/
    connections: List[Dict[str, Any]] = []

    # From /api/v1/builder/node-types
    node_types: Dict[str, Any] = {}

    # From /api/v1/workflows
    workflows: Dict[str, Any] = {"items": [], "next_cursor": None}

    # Metadata
    cached: bool = False
    timestamp: Optional[str] = None
