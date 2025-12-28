"""
Example workflow that reads the latest three Gmail messages and summarizes them
with an LLM.

Usage:
    OPENAI_API_KEY=... python -m workflow_compiler.examples.gmail_summary_workflow

Set USE_REAL_GMAIL=1 (and ensure OAuth credentials exist for user 1) to hit the
live Gmail API; otherwise the script uses deterministic stub data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from workflow_compiler import compile_workflow
from workflow_compiler.compiler.context import CompilerContext
from workflow_compiler.examples.gmail_common import GmailDemoService, fetch_oauth_credentials
from workflow_compiler.registry.model_registry import ModelDefinition, ModelRegistry
from workflow_compiler.registry.tool_registry import ToolDefinition, ToolRegistry
from workflow_compiler.schema.models import JsonSchema
from workflow_compiler.schema.schema_registry import SchemaRegistry
from shared.tools.google.gmail import GmailReadTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_schema_registry() -> SchemaRegistry:
    return SchemaRegistry()


def build_tool_registry(service: GmailDemoService) -> ToolRegistry:
    read_tool = GmailReadTool()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="gmail_read_emails",
            version="v1",
            input_schema=read_tool.get_parameters_schema(),
            output_schema=read_tool.get_output_schema(),
            handler=lambda params, config: service.read_emails(params),
        )
    )
    return registry


def build_model_registry() -> ModelRegistry:
    try:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
            temperature=0.3,
        )
    except Exception as exc:  # pragma: no cover - depends on env configuration
        raise RuntimeError(
            "ChatOpenAI initialization failed. Ensure OPENAI_API_KEY is configured."
        ) from exc

    def llm_handler(invocation: Dict[str, Any]) -> str:
        emails = invocation.get("inputs", {}).get("emails", [])
        rendered_prompt = (
            f"{invocation['prompt']}\n\n"
            "Emails JSON:\n"
            f"{json.dumps(emails, indent=2)}\n\n"
            "Provide a concise numbered summary highlighting key senders, topics, and action items."
        )
        response = llm.invoke(rendered_prompt)
        return response.content if hasattr(response, "content") else str(response)

    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            model_id="gmail-summary-llm",
            text_handler=llm_handler,
        )
    )
    return registry


def build_workflow_spec(email_schema: JsonSchema) -> Dict[str, Any]:
    return {
        "version": "1",
        "inputs": {
            "user_id": {"type": "integer", "required": True, "description": "Owner of the Gmail mailbox"},
        },
        "nodes": [
            {
                "id": "fetch_emails",
                "type": "tool",
                "tool": "gmail_read_emails",
                "in": {
                    "user_id": "${inputs.user_id}",
                    "max_results": 3,
                    "label_ids": ["INBOX"],
                    "include_body": True,
                },
                "out": "emails",
                "expect_output": {
                    "mode": "json",
                    "schema": {"json_schema": email_schema},
                },
            },
            {
                "id": "summarize",
                "type": "llm",
                "model": "gmail-summary-llm",
                "prompt": (
                    "Summarize the following emails for a busy engineer. "
                    "Group similar items and call out any requests or deadlines."
                ),
                "in": {"emails": "${emails}"},
                "out": "inbox_summary",
                "output": {"mode": "text"},
            },
        ],
        "output": "${inbox_summary}",
    }


def main() -> None:
    user_id = 1
    credentials = asyncio.run(fetch_oauth_credentials(user_id))
    service = GmailDemoService(credentials)

    read_schema = service.read_tool.get_output_schema()
    schema_registry = build_schema_registry()
    tool_registry = build_tool_registry(service)
    model_registry = build_model_registry()

    spec = build_workflow_spec(read_schema)
    context = CompilerContext(
        schema_registry=schema_registry,
        tool_registry=tool_registry,
        model_registry=model_registry,
    )

    compiled = compile_workflow(spec, context)
    summary = compiled.invoke(inputs={"user_id": user_id})
    print("Inbox summary:\n", summary.get("inbox_summary"))


if __name__ == "__main__":
    main()


