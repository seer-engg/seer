"""
Helpers for surfacing the canonical workflow compiler schema to the agent.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from seer.core.schema.models import (
    WorkflowSpec,
    EdgeType,
    ToolNode,
    LLMNode,
    TaskNode,
    IfNode,
    ForEachNode,
    TaskKind,
    TriggerSpec,
    Edge,
)
from seer.logger import get_logger

logger = get_logger(__name__)

# Keys that keep the schema digestible while conveying the structure.
_SCHEMA_KEYS = ("title", "type", "properties", "required", "definitions", "default")

_WORKFLOW_SPEC_EXAMPLE: Dict[str, Any] = {
    "version": "2",
    "triggers": [],
    "nodes": [
        {
            "id": "fetch_news",
            "type": "tool",
            "tool": "demo.news_search",
            "inputs": {
                "query": "AI automation trends",
                "timeframe_days": 7,
            },
        },
        {
            "id": "summarize",
            "type": "llm",
            "inputs": {
                "model": "gpt-4o-mini",
                "prompt": "Summarize the top 3 recent articles. Use bullet points with source names.",
                "articles": "${fetch_news}",
            },
            "outputs": {
                "mode": "json",
                "schema": {
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "talking_points": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["talking_points"],
                    }
                },
            },
        },
    ],
    "edges": [
        {"source": "fetch_news", "target": "summarize"},
    ],
}

_WORKFLOW_SPEC_TRIGGER_EXAMPLE: Dict[str, Any] = {
    "version": "2",
    "triggers": [
        {
            "id": "new_signup",
            "key": "webhook.supabase.db_changes",
            "mode": "webhook",
            "provider_config": {
                "integration_resource_id": 123,
                "table": "signups",
                "schema": "public",
                "events": ["INSERT"]
            },
            "event_schema": {
                "type": "object",
                "properties": {
                    "record": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "name": {"type": "string"}
                        }
                    }
                },
                "required": ["record"]
            }
        }
    ],
    "nodes": [
        {
            "id": "extract_user",
            "type": "task",
            "kind": "set",
            "value": "${trigger.data.record}",
        },
        {
            "id": "create_draft",
            "type": "tool",
            "tool": "gmail_create_draft",
            "inputs": {
                "to": ["${extract_user.email}"],
                "subject": "Welcome!",
                "body_text": "Hi ${extract_user.name}, welcome to our platform!"
            },
        }
    ],
    "edges": [
        {"source": "new_signup", "target": "extract_user", "type": "trigger"},
        {"source": "extract_user", "target": "create_draft"},
    ],
}


@lru_cache(maxsize=1)
def get_workflow_spec_schema() -> Dict[str, Any]:
    """
    Return the compiler JSON schema for WorkflowSpec with only the most relevant keys.
    Cached because pydantic schema generation is relatively expensive.
    """

    schema = WorkflowSpec.model_json_schema()
    return {key: schema.get(key) for key in _SCHEMA_KEYS if key in schema}


def get_workflow_spec_schema_text(max_chars: int = 4000) -> str:
    """
    Render the schema as formatted JSON, optionally truncating for prompt safety.
    """

    schema_text = json.dumps(get_workflow_spec_schema(), indent=2)
    if len(schema_text) > max_chars:
        schema_text = schema_text[: max_chars - 3] + "..."
    return schema_text


def get_workflow_spec_example_text() -> str:
    """
    Provide compact, valid WorkflowSpec examples for the agent to imitate.
    Includes both input-based and trigger-based workflow examples.
    """

    examples_text = "Example 1 (Input-based workflow):\n"
    examples_text += json.dumps(_WORKFLOW_SPEC_EXAMPLE, indent=2)
    examples_text += "\n\nExample 2 (Trigger-based workflow):\n"
    examples_text += json.dumps(_WORKFLOW_SPEC_TRIGGER_EXAMPLE, indent=2)

    return examples_text


@lru_cache(maxsize=1)
def get_workflow_templates() -> List[Dict[str, Any]]:
    """
    Load all workflow templates from the templates directory.
    Templates provide common workflow patterns that the agent can suggest or use as starting points.

    Returns:
        List of template dictionaries with name, description, tags, customization_guide, and spec
    """
    templates_dir = Path(__file__).parent / "templates"
    templates = []

    if not templates_dir.exists():
        logger.warning("Templates directory not found at %s", templates_dir)
        return templates

    for template_file in templates_dir.glob("*.json"):
        try:
            with open(template_file, "r", encoding="utf-8") as file:
                template_data = json.load(file)
                templates.append(template_data)
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Failed to load template %s: %s", template_file.name, exc)
            continue

    logger.info("Loaded %d workflow templates", len(templates))
    return templates


def get_workflow_templates_summary() -> str:
    """
    Generate a concise summary of available workflow templates for the agent system prompt.

    Returns:
        Formatted string listing templates with their descriptions and use cases
    """
    templates = get_workflow_templates()

    if not templates:
        return "No workflow templates available."

    summary_lines = ["## Common Workflow Templates\n"]
    summary_lines.append("You can suggest these templates when they match user intent:\n")

    for idx, template in enumerate(templates, 1):
        name = template.get("name", "Unknown")
        description = template.get("description", "")
        tags = template.get("tags", [])

        summary_lines.append(f"{idx}. **{name}**")
        summary_lines.append(f"   - Description: {description}")
        summary_lines.append(f"   - Use when: {', '.join(tags[:4])}")
        summary_lines.append("")

    return "\n".join(summary_lines)


def _format_node_type_usage_notes(node_type: str) -> List[str]:
    """Generate usage notes for a specific node type."""
    notes_map = {
        "tool": [
            "**Important:** Tool nodes do NOT have an `outputs` field. "
            "Tool output schema is determined by the tool registry."
        ],
        "llm": [
            "**Important:** LLM nodes require `model` and `prompt` in inputs. "
            "Use `outputs` to specify structured JSON output with schema."
        ],
        "task": [
            f"**Task kinds:** {', '.join(k.value for k in TaskKind)}",
            "**Important:** `kind=set` requires `value` field. "
            "Use variable references like `${node_id}` to access node outputs."
        ],
        "if": [
            "**Important:** Condition should be a boolean expression. "
            "Use edges with `type=conditional_true` and `type=conditional_false` for branching."
        ],
        "for_each": [
            "**Important:** Items should evaluate to a list. "
            "Use edges with `type=loop_body` for loop content and `type=loop_exit` to exit."
        ],
    }
    return notes_map.get(node_type, [])


@lru_cache(maxsize=1)
def generate_node_type_reference() -> str:
    """
    Auto-generate node type reference from Pydantic models.

    Extracts documentation from ToolNode, LLMNode, TaskNode, IfNode, ForEachNode
    including required fields, all fields with descriptions, and usage notes.

    Returns:
        Formatted documentation for each node type
    """
    node_types = {
        "tool": (ToolNode, "Execute a tool from the tool registry"),
        "llm": (LLMNode, "AI inference with model configuration"),
        "task": (TaskNode, "Data transformation and manipulation"),
        "if": (IfNode, "Conditional branching based on expression"),
        "for_each": (ForEachNode, "Iterate over a list with loop body"),
    }

    lines = []

    for node_type, (model_cls, description) in node_types.items():
        schema = model_cls.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        lines.append(f"### {node_type}")
        lines.append(description)
        lines.append("")
        lines.append("**Required fields:**")

        for field_name in required:
            if field_name in properties:
                prop = properties[field_name]
                field_type = prop.get("type", "any")
                field_desc = prop.get("description", "")
                default = prop.get("default", "")
                default_str = f" (default: {json.dumps(default)})" if default else ""
                lines.append(f"- `{field_name}` ({field_type}){default_str}: {field_desc}")

        lines.append("")
        lines.append("**All fields:**")

        for field_name, prop in properties.items():
            if field_name in ["type", "ui"]:
                continue
            field_type = prop.get("type", prop.get("anyOf", "any"))
            field_desc = prop.get("description", "")
            is_required = field_name in required
            req_str = " (required)" if is_required else " (optional)"

            if isinstance(field_type, list):
                field_type = "/".join(str(t) for t in field_type)
            elif isinstance(field_type, dict):
                field_type = "object"

            lines.append(f"- `{field_name}`{req_str}: {field_desc}")

        usage_notes = _format_node_type_usage_notes(node_type)
        if usage_notes:
            lines.append("")
            lines.extend(usage_notes)

        lines.append("")

    return "\n".join(lines)


@lru_cache(maxsize=1)
def generate_validation_checklist_from_model() -> str:
    """
    Auto-generate validation checklist from WorkflowSpec Pydantic model.

    Extracts requirements from model_json_schema() to create a checklist of
    validation rules for workflow generation.

    Returns:
        Checklist items extracted from schema
    """
    spec_schema = WorkflowSpec.model_json_schema()
    trigger_schema = TriggerSpec.model_json_schema()
    edge_schema = Edge.model_json_schema()

    lines = ["**Workflow Validation Checklist:**", ""]

    spec_props = spec_schema.get("properties", {})

    lines.append("Top-level requirements:")
    lines.append(f"- version must be \"2\" (default: {spec_props.get('version', {}).get('default', 'N/A')})")
    lines.append("- nodes must be a list of node objects")
    lines.append("- edges must be a list of edge objects")
    lines.append("- triggers must be a list of trigger objects")
    lines.append("")

    lines.append("Node requirements:")
    lines.append("- Each node must have unique `id` (string, min 1 char)")
    lines.append("- Each node must have `type` field: tool, llm, task, if, or for_each")
    lines.append("- Tool nodes: require `tool` (string) and `inputs` (object)")
    lines.append("- LLM nodes: require `inputs` with `model` and `prompt` fields")
    lines.append("- Task nodes: require `kind` (set) and `value` if kind=set")
    lines.append("- If nodes: require `condition` (boolean expression)")
    lines.append("- ForEach nodes: require `items` (list expression)")
    lines.append("- Tool nodes MUST NOT have `outputs` field (derived from registry)")
    lines.append("- Node outputs accessed via variable syntax: `${node_id}` or `${node_id.field}`")
    lines.append("")

    trigger_props = trigger_schema.get("properties", {})
    trigger_required = trigger_schema.get("required", [])

    lines.append("Trigger requirements:")
    for field in trigger_required:
        if field in trigger_props:
            prop = trigger_props[field]
            field_desc = prop.get("description", "")
            lines.append(f"- {field}: {field_desc}")
    lines.append("")

    edge_props = edge_schema.get("properties", {})
    edge_required = edge_schema.get("required", [])

    lines.append("Edge requirements:")
    for field in edge_required:
        if field in edge_props:
            prop = edge_props[field]
            field_desc = prop.get("description", "")
            lines.append(f"- {field}: {field_desc}")

    edge_types = [e.value for e in EdgeType]
    lines.append(f"- Valid edge types: {', '.join(edge_types)}")
    lines.append("  - default: sequential flow")
    lines.append("  - trigger: from trigger to first node")
    lines.append("  - conditional_true/conditional_false: if node branches")
    lines.append("  - loop_body/loop_exit: for_each loop edges")
    lines.append("")

    return "\n".join(lines)


@lru_cache(maxsize=1)
def generate_trigger_reference() -> str:
    """
    Auto-generate trigger documentation from TriggerSpec model.

    Returns:
        Formatted trigger specification documentation
    """
    schema = TriggerSpec.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    lines = ["### Trigger Specification", ""]
    lines.append("Triggers define when workflows execute. Each trigger must have:")
    lines.append("")

    lines.append("**Required fields:**")
    for field_name in required:
        if field_name in properties:
            prop = properties[field_name]
            field_desc = prop.get("description", "")
            lines.append(f"- `{field_name}`: {field_desc}")

    lines.append("")
    lines.append("**Configuration fields:**")
    for field_name, prop in properties.items():
        if field_name in required or field_name in ["ui_meta"]:
            continue
        field_desc = prop.get("description", "")
        lines.append(f"- `{field_name}`: {field_desc}")

    lines.append("")
    lines.append("**Important:**")
    lines.append("- Trigger `id` must be unique across all triggers in the workflow")
    lines.append("- Use edge with `type=trigger` to connect trigger to first node")
    lines.append("- Trigger data accessed via `${trigger.data}` in downstream nodes")
    lines.append("- Provider-specific config goes in `provider_config` object")
    lines.append("")

    return "\n".join(lines)


@lru_cache(maxsize=1)
def generate_edge_reference() -> str:
    """
    Auto-generate edge type documentation from Edge and EdgeType models.

    Returns:
        Formatted edge specification documentation
    """
    schema = Edge.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    lines = ["### Edge Specification", ""]
    lines.append("Edges connect nodes to define workflow execution flow.")
    lines.append("")

    lines.append("**Required fields:**")
    for field_name in required:
        if field_name in properties:
            prop = properties[field_name]
            field_desc = prop.get("description", "")
            lines.append(f"- `{field_name}`: {field_desc}")

    lines.append("")
    lines.append("**Edge types:**")
    for edge_type in EdgeType:
        lines.append(f"- `{edge_type.value}`: {_get_edge_type_description(edge_type)}")

    lines.append("")
    lines.append("**Important:**")
    lines.append("- Every node must be reachable via edges (no orphaned nodes)")
    lines.append("- Trigger nodes connect to first processing node via `type=trigger` edge")
    lines.append("- If nodes must have both conditional_true and conditional_false edges")
    lines.append("- ForEach nodes must have loop_body and loop_exit edges")
    lines.append("")

    return "\n".join(lines)


def _get_edge_type_description(edge_type: EdgeType) -> str:
    """Helper to provide human-readable descriptions for edge types."""
    descriptions = {
        EdgeType.default: "Sequential flow from source to target",
        EdgeType.trigger: "Entry point from trigger to first node",
        EdgeType.conditional_true: "If condition evaluates to true",
        EdgeType.conditional_false: "If condition evaluates to false",
        EdgeType.loop_body: "Entry into for_each loop body",
        EdgeType.loop_exit: "Exit from for_each loop",
    }
    return descriptions.get(edge_type, "Unknown edge type")
