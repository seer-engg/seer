"""
Helpers for surfacing the canonical workflow compiler schema to the agent.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from seer.core.schema.models import WorkflowSpec
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
