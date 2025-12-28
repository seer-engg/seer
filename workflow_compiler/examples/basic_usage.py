"""
Minimal end-to-end example that compiles and runs a workflow using the
workflow_compiler package.

Run with:

    python -m workflow_compiler.examples.basic_usage
"""

from __future__ import annotations

from typing import Any, Dict

from workflow_compiler import compile_workflow
from workflow_compiler.compiler.context import CompilerContext
from workflow_compiler.registry.model_registry import ModelDefinition, ModelRegistry
from workflow_compiler.registry.tool_registry import ToolDefinition, ToolRegistry
from workflow_compiler.schema.schema_registry import SchemaRegistry


def build_context() -> CompilerContext:
    """
    Constructs registries with minimal demo implementations.
    """

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

    schema_registry = SchemaRegistry(
        {
            "schemas.issue_results@v1": issue_schema,
            "schemas.summary@v1": summary_schema,
        }
    )

    tool_registry = ToolRegistry()
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

    model_registry = ModelRegistry()
    model_registry.register(
        ModelDefinition(
            model_id="demo-text-model",
            handler=_run_llm,
            supports_structured_output=False,
        )
    )

    return CompilerContext(
        schema_registry=schema_registry,
        tool_registry=tool_registry,
        model_registry=model_registry,
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


def _run_llm(prompt: str, request: Dict[str, Any]) -> Dict[str, Any]:
    return {"message": f"[LLM SUMMARY]\\n{prompt}"}


def build_workflow_spec() -> Dict[str, Any]:
    """
    Returns a simple workflow identical to the one described in the design doc.
    """

    return {
        "version": "1",
        "inputs": {
            "repo": {"type": "string", "required": True},
            "query": {"type": "string", "required": True},
        },
        "nodes": [
            {
                "id": "search_issue",
                "type": "tool",
                "tool": "github.search_issues",
                "in": {"repo": "${inputs.repo}", "q": "${inputs.query}"},
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
    context = build_context()
    spec = build_workflow_spec()
    compiled = compile_workflow(spec, context)
    result = compiled.invoke(
        inputs={
            "repo": "seer-engg/seer",
            "query": "regression",
        }
    )
    print("Workflow result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()


