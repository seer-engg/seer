"""
Minimal end-to-end example that compiles and runs a workflow using the
workflow_compiler package.

Run with:

    python -m workflow_compiler.examples.basic_usage
"""

from __future__ import annotations

from typing import Any, Dict

from seer.database.models import User
from seer.core.registry.model_registry import ModelDefinition, ModelRegistry
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.runtime import WorkflowCompilerSingleton
from seer.core.schema.schema_registry import SchemaRegistry


def register_demo_components() -> None:
    """
    Register demo schemas, tools, and models in the singleton compiler so the
    example workflow can compile without custom contexts.
    """

    compiler = WorkflowCompilerSingleton.instance()

    issue_schema = {
        "type": "object",
        "properties": {
            "total_count": {"type": "integer"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["title", "body"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["total_count", "items"],
        "additionalProperties": False,
    }
    summary_schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
        },
        "required": ["message"],
        "additionalProperties": False,
    }

    schema_registry: SchemaRegistry = compiler._schema_registry  # type: ignore[attr-defined]
    if not schema_registry.has_schema("schemas.issue_results@v1"):
        schema_registry.register("schemas.issue_results@v1", issue_schema)
    if not schema_registry.has_schema("schemas.summary@v1"):
        schema_registry.register("schemas.summary@v1", summary_schema)

    tool_registry: ToolRegistry = compiler._tool_registry  # type: ignore[attr-defined]
    if not tool_registry.maybe_get("github.search_issues"):
        tool_registry.register(
            ToolDefinition(
                name="github.search_issues",
                version="v1",
                input_schema={
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string"},
                        "q": {"type": "string"},
                    },
                    "required": ["repo", "q"],
                },
                output_schema=issue_schema,
                handler=_search_issues,
            )
        )

    model_registry: ModelRegistry = compiler._model_registry  # type: ignore[attr-defined]
    if not model_registry.maybe_get("demo-text-model"):
        model_registry.register(
            ModelDefinition(
                model_id="demo-text-model",
                json_handler=_run_llm,
            )
        )


def _search_issues(inputs: Dict[str, Any], config: Dict[str, Any] | None) -> Dict[str, Any]:
    repo = inputs["repo"]
    query = inputs["q"]
    return {
        "total_count": 1,
        "items": [
            {
                "title": f"{repo} :: {query}",
                "body": "Example issue body explaining the bug in detail.",
            }
        ],
    }


def _run_llm(invocation: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    return {"message": f"[LLM SUMMARY]\\n{invocation['prompt']}"}


def build_workflow_spec() -> Dict[str, Any]:
    """
    Returns a simple workflow identical to the one described in the design doc.
    """

    return {
        "version": "2",
        "triggers": [
            {
                "key": "manual.trigger",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "repo": {"type": "string"},
                                    "query": {"type": "string"},
                                },
                                "required": ["repo", "query"],
                            }
                        },
                    }
                },
            }
        ],
        "edges": [
            {"id": "e0", "source": "manual.trigger", "target": "search_issue", "type": "trigger"}
        ],
        "nodes": [
            {
                "id": "search_issue",
                "type": "tool",
                "tool": "github.search_issues",
                "in": {"repo": "${trigger.data.repo}", "q": "${trigger.data.query}"},
                "out": "issue_search",
                "expect_output": {
                    "mode": "json",
                    "schema": {"id": "schemas.issue_results@v1"},
                },
            },
            {
                "id": "route",
                "type": "if",
                "condition": "${issue_search.total_count} > 0",
                "then": [
                    {
                        "id": "summarize",
                        "type": "llm",
                        "model": "demo-text-model",
                        "prompt": "Summarize issue: ${issue_search.items[0].title}",
                        "out": "summary",
                        "output": {
                            "mode": "json",
                            "schema": {"id": "schemas.summary@v1"},
                        },
                    }
                ],
                "else": [
                    {
                        "id": "no_results",
                        "type": "task",
                        "kind": "set",
                        "value": {"message": "No issues found"},
                        "out": "summary",
                        "output": {
                            "mode": "json",
                            "schema": {"id": "schemas.summary@v1"},
                        },
                    }
                ],
            },
        ],
        "output": "${summary}",
    }


def main() -> None:
    register_demo_components()
    compiler = WorkflowCompilerSingleton.instance()
    demo_user = User(id=0, user_id="demo-basic-user")
    spec = build_workflow_spec()
    compiled = compiler.compile(demo_user, spec)
    result = compiled.invoke(
        trigger={
            "trigger_key": "manual.trigger",
            "data": {
                "repo": "seer-engg/seer",
                "query": "regression",
            },
        }
    )
    print("Workflow result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
