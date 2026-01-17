"""Workflow management tools for MCP server."""
# pylint: disable=duplicate-code # Reason: Similar inputSchema patterns for API consistency

import json
from typing import Any, Dict, List, Optional

from mcp.types import TextContent, Tool  # pylint: disable=no-name-in-module # Reason: external MCP SDK

from api.workflows import models as api_models
from api.workflows.services.lifecycle import (
    create_workflow,
    list_workflows,
    get_workflow,
    patch_workflow_draft,
    delete_workflow,
)
from seer_mcp.config import get_config, MCPMode
from shared.database import User


async def seer_create_workflow(
    user: User,
    name: str,
    description: str = "",
    spec: Optional[Dict[str, Any]] = None,
) -> List[TextContent]:
    """Create a new workflow.

    Args:
        user: Authenticated user
        name: Workflow name
        description: Optional workflow description
        spec: Optional workflow specification

    Returns:
        List of text content with workflow creation result
    """
    config = get_config()

    if spec is None:
        spec = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "name": name,
                "description": description,
            },
        }

    if config.mode == MCPMode.LOCAL:
        # LOCAL mode: Direct service call
        payload = api_models.WorkflowCreateRequest(
            name=name,
            description=description,
            spec=spec,
        )
        result = await create_workflow(user, payload)
        result_dict = {
            "workflow_id": result.workflow_id,
            "name": result.name,
            "description": result.description,
            "draft_revision": result.draft_revision,
            "created_at": str(result.created_at),
        }
    else:
        # CLOUD mode: API client call
        from seer_mcp.server import get_api_client  # pylint: disable=import-outside-toplevel,cyclic-import # Reason: Avoid circular import

        client = get_api_client()
        if not client:
            raise ValueError("API client not initialized")
        result_dict = await client.create_workflow(name, description, spec)

    return [
        TextContent(
            type="text",
            text=f"""Created workflow successfully!

Workflow ID: {result_dict['workflow_id']}
Name: {result_dict['name']}
Description: {result_dict.get('description') or "No description"}
Draft Revision: {result_dict['draft_revision']}
Created At: {result_dict['created_at']}

You can now add nodes and edges to the workflow spec or run it in test mode.""",
        )
    ]


async def seer_list_workflows(
    user: User,
    limit: int = 20,
    cursor: Optional[str] = None,
) -> List[TextContent]:
    """List user's workflows.

    Args:
        user: Authenticated user
        limit: Maximum number of workflows to return (default: 20, max: 100)
        cursor: Optional cursor for pagination

    Returns:
        List of text content with workflows
    """
    config = get_config()

    if limit < 1 or limit > 100:
        limit = min(max(limit, 1), 100)

    if config.mode == MCPMode.LOCAL:
        # LOCAL mode: Direct service call
        result = await list_workflows(user, limit=limit, cursor=cursor)
        workflows = result.workflows
        next_cursor = result.next_cursor
    else:
        # CLOUD mode: API client call
        from seer_mcp.server import get_api_client  # pylint: disable=import-outside-toplevel,cyclic-import # Reason: Avoid circular import

        client = get_api_client()
        if not client:
            raise ValueError("API client not initialized")
        result_dict = await client.list_workflows(limit=limit, cursor=cursor)
        workflows = result_dict.get("items", [])
        next_cursor = result_dict.get("next_cursor")

    if not workflows:
        return [TextContent(type="text", text="No workflows found.")]

    workflows_text = "\n\n".join(
        [
            f"""Workflow ID: {w.get('workflow_id') if isinstance(w, dict) else w.workflow_id}
Name: {w.get('name') if isinstance(w, dict) else w.name}
Description: {(w.get('description') if isinstance(w, dict) else w.description) or "No description"}
Draft Revision: {w.get('draft_revision') if isinstance(w, dict) else w.draft_revision}
Created: {w.get('created_at') if isinstance(w, dict) else w.created_at}
Updated: {w.get('updated_at') if isinstance(w, dict) else w.updated_at}"""
            for w in workflows
        ]
    )

    pagination_info = f"\n\nNext cursor: {next_cursor}" if next_cursor else ""

    return [
        TextContent(
            type="text",
            text=f"""Found {len(workflows)} workflow(s):

{workflows_text}{pagination_info}""",
        )
    ]


async def seer_get_workflow(
    user: User,
    workflow_id: str,
) -> List[TextContent]:
    """Get workflow details by ID.

    Args:
        user: Authenticated user
        workflow_id: Workflow ID (e.g., "wf_123")

    Returns:
        List of text content with workflow details
    """
    config = get_config()

    if config.mode == MCPMode.LOCAL:
        # LOCAL mode: Direct service call
        result = await get_workflow(user, workflow_id)
        spec_dict = result.spec.model_dump()
        result_dict = {
            "workflow_id": result.workflow_id,
            "name": result.name,
            "description": result.description,
            "draft_revision": result.draft_revision,
            "created_at": str(result.created_at),
            "updated_at": str(result.updated_at),
            "spec": spec_dict,
            "published_version": result.published_version.model_dump() if result.published_version else None,
            "latest_version": result.latest_version.model_dump() if result.latest_version else None,
        }
    else:
        # CLOUD mode: API client call
        from seer_mcp.server import get_api_client  # pylint: disable=import-outside-toplevel,cyclic-import # Reason: Avoid circular import

        client = get_api_client()
        if not client:
            raise ValueError("API client not initialized")
        result_dict = await client.get_workflow(workflow_id)

    spec_json = json.dumps(result_dict["spec"], indent=2)

    version_info = ""
    if result_dict.get("published_version"):
        version_info += f"\nPublished Version: v{result_dict['published_version'].get('version_number')}"
    if result_dict.get("latest_version"):
        version_info += f"\nLatest Version: v{result_dict['latest_version'].get('version_number')}"

    return [
        TextContent(
            type="text",
            text=f"""Workflow Details:

ID: {result_dict['workflow_id']}
Name: {result_dict['name']}
Description: {result_dict.get('description') or "No description"}
Draft Revision: {result_dict['draft_revision']}
Created: {result_dict['created_at']}
Updated: {result_dict['updated_at']}{version_info}

Workflow Spec:
{spec_json}""",
        )
    ]


async def seer_update_workflow(
    user: User,
    workflow_id: str,
    spec: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> List[TextContent]:
    """Update a workflow's spec, name, or description.

    Args:
        user: Authenticated user
        workflow_id: Workflow ID (e.g., "wf_123")
        spec: Optional new workflow specification
        name: Optional new name
        description: Optional new description

    Returns:
        List of text content with update result
    """
    config = get_config()

    if spec is None and name is None and description is None:
        return [
            TextContent(
                type="text",
                text="No updates provided. Please specify spec, name, or description to update.",
            )
        ]

    if config.mode == MCPMode.LOCAL:
        # LOCAL mode: Direct service call
        payload_dict: Dict[str, Any] = {}
        if spec is not None:
            payload_dict["spec"] = spec
        if name is not None:
            payload_dict["name"] = name
        if description is not None:
            payload_dict["description"] = description

        payload = api_models.WorkflowUpdateRequest(**payload_dict)
        result = await patch_workflow_draft(user, workflow_id, payload)
        result_dict = {
            "name": result.name,
            "description": result.description,
            "draft_revision": result.draft_revision,
            "updated_at": str(result.updated_at),
        }
    else:
        # CLOUD mode: API client call
        from seer_mcp.server import get_api_client  # pylint: disable=import-outside-toplevel,cyclic-import # Reason: Avoid circular import

        client = get_api_client()
        if not client:
            raise ValueError("API client not initialized")
        result_dict = await client.update_workflow(workflow_id, name, description, spec)

    return [
        TextContent(
            type="text",
            text=f"""Updated workflow {workflow_id} successfully!

Name: {result_dict['name']}
Description: {result_dict.get('description') or "No description"}
Draft Revision: {result_dict['draft_revision']}
Updated At: {result_dict['updated_at']}""",
        )
    ]


async def seer_delete_workflow(
    user: User,
    workflow_id: str,
) -> List[TextContent]:
    """Delete a workflow.

    Args:
        user: Authenticated user
        workflow_id: Workflow ID (e.g., "wf_123")

    Returns:
        List of text content with deletion result
    """
    config = get_config()

    if config.mode == MCPMode.LOCAL:
        # LOCAL mode: Direct service call
        await delete_workflow(user, workflow_id)
    else:
        # CLOUD mode: API client call
        from seer_mcp.server import get_api_client  # pylint: disable=import-outside-toplevel,cyclic-import # Reason: Avoid circular import

        client = get_api_client()
        if not client:
            raise ValueError("API client not initialized")
        await client.delete_workflow(workflow_id)

    return [
        TextContent(
            type="text",
            text=f"Deleted workflow {workflow_id} successfully.",
        )
    ]


def get_workflow_tools() -> List[Tool]:
    """Get workflow management tool definitions for MCP."""
    return [
        Tool(
            name="seer_create_workflow",
            description="Create a new workflow in Seer",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Workflow name",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional workflow description",
                    },
                    "spec": {
                        "type": "object",
                        "description": "Optional workflow specification with nodes and edges",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="seer_list_workflows",
            description="List all workflows for the authenticated user",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of workflows to return (1-100, default: 20)",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Optional cursor for pagination",
                    },
                },
            },
        ),
        Tool(
            name="seer_get_workflow",
            description="Get detailed information about a specific workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID (e.g., 'wf_123')",
                    },
                },
                "required": ["workflow_id"],
            },
        ),
        Tool(
            name="seer_update_workflow",
            description="Update a workflow's specification, name, or description",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID (e.g., 'wf_123')",
                    },
                    "spec": {
                        "type": "object",
                        "description": "Optional new workflow specification",
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional new workflow name",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional new workflow description",
                    },
                },
                "required": ["workflow_id"],
            },
        ),
        Tool(
            name="seer_delete_workflow",
            description="Delete a workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID to delete (e.g., 'wf_123')",
                    },
                },
                "required": ["workflow_id"],
            },
        ),
    ]
