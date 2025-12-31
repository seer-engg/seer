"""
Helpers for surfacing the canonical workflow compiler schema to the agent.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict

from workflow_compiler.schema.models import WorkflowSpec

# Keys that keep the schema digestible while conveying the structure.
_SCHEMA_KEYS = ("title", "type", "properties", "required", "definitions")

_WORKFLOW_SPEC_EXAMPLE: Dict[str, Any] = {
    "version": "1",
    "inputs": {
        "company": {
            "type": "string",
            "description": "Company name we are researching",
            "required": True,
        }
    },
    "nodes": [
        {
            "id": "fetch_news",
            "type": "tool",
            "tool": "demo.news_search",
            "in": {
                "query": "${inputs.company}",
                "timeframe_days": 7,
            },
            "out": "news_results",
            "expect_output": {
                "mode": "json",
                "schema": {
                    "json_schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "summary": {"type": "string"},
                            },
                            "required": ["title", "url"],
                        },
                    }
                },
            },
        },
        {
            "id": "summarize",
            "type": "llm",
            "model": "gpt-5-mini",
            "prompt": (
                "Summarize the top 3 recent articles about ${inputs.company}. "
                "Use bullet points with source names."
            ),
            "in": {"articles": "${news_results}"},
            "out": "company_summary",
            "output": {
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
    "output": "${company_summary}",
    "meta": {
        "description": "Fetch latest company news and summarize key talking points."
    },
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
    Provide a compact, valid WorkflowSpec example for the agent to imitate.
    """

    return json.dumps(_WORKFLOW_SPEC_EXAMPLE, indent=2)

