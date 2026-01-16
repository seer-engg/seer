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
    update_workflow_draft,
    delete_workflow,
)
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
    if spec is None:
        spec = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "name": name,
                "description": description,
            },
        }

    payload = api_models.WorkflowCreateRequest(
        name=name,
        description=description,
        spec=spec,
    )

    result = await create_workflow(user, payload)

    return [
        TextContent(
            type="text",
            text=f"""Created workflow successfully!

Workflow ID: {result.workflow_id}
Name: {result.name}
Description: {result.description or "No description"}
Draft Revision: {result.draft_revision}
Created At: {result.created_at}

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
    if limit < 1 or limit > 100:
        limit = min(max(limit, 1), 100)

    result = await list_workflows(user, limit=limit, cursor=cursor)

    if not result.workflows:
        return [TextContent(type="text", text="No workflows found.")]

    workflows_text = "\n\n".join(
        [
            f"""Workflow ID: {w.workflow_id}
Name: {w.name}
Description: {w.description or "No description"}
Draft Revision: {w.draft_revision}
Created: {w.created_at}
Updated: {w.updated_at}"""
            for w in result.workflows
        ]
    )

    pagination_info = f"\n\nNext cursor: {result.next_cursor}" if result.next_cursor else ""

    return [
        TextContent(
            type="text",
            text=f"""Found {len(result.workflows)} workflow(s):

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
    result = await get_workflow(user, workflow_id)

    spec_json = json.dumps(result.spec.model_dump(), indent=2)

    version_info = ""
    if result.published_version:
        version_info += f"\nPublished Version: v{result.published_version.version_number}"
    if result.latest_version:
        version_info += f"\nLatest Version: v{result.latest_version.version_number}"

    return [
        TextContent(
            type="text",
            text=f"""Workflow Details:

ID: {result.workflow_id}
Name: {result.name}
Description: {result.description or "No description"}
Draft Revision: {result.draft_revision}
Created: {result.created_at}
Updated: {result.updated_at}{version_info}

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
    payload_dict: Dict[str, Any] = {}

    if spec is not None:
        payload_dict["spec"] = spec
    if name is not None:
        payload_dict["name"] = name
    if description is not None:
        payload_dict["description"] = description

    if not payload_dict:
        return [
            TextContent(
                type="text",
                text="No updates provided. Please specify spec, name, or description to update.",
            )
        ]

    payload = api_models.WorkflowUpdateRequest(**payload_dict)
    result = await update_workflow_draft(user, workflow_id, payload)

    return [
        TextContent(
            type="text",
            text=f"""Updated workflow {workflow_id} successfully!

Name: {result.name}
Description: {result.description or "No description"}
Draft Revision: {result.draft_revision}
Updated At: {result.updated_at}""",
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
    await delete_workflow(user, workflow_id)

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
