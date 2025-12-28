"""
Example workflow that:

1. Reads the most recent Gmail message.
2. Drafts an LLM reply.
3. Creates a Gmail draft with the generated reply.

Usage:
    OPENAI_API_KEY=... python -m workflow_compiler.examples.gmail_reply_workflow

By default the script stubs Gmail interactions so it can run without network
access. If you have valid OAuth credentials in PostgreSQL (see
shared/database/models_oauth.py) and want to hit the Gmail API for real, set:

    USE_REAL_GMAIL=1
"""

from __future__ import annotations

import asyncio
import logging
import os
import json
from typing import Any, Dict, List, Mapping

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from workflow_compiler import compile_workflow
from workflow_compiler.compiler.context import CompilerContext
from workflow_compiler.registry.model_registry import ModelDefinition, ModelRegistry
from workflow_compiler.registry.tool_registry import ToolDefinition, ToolRegistry
from workflow_compiler.schema.schema_registry import SchemaRegistry
from workflow_compiler.schema.models import JsonSchema

from workflow_compiler.examples.gmail_common import (
    GmailDemoService,
    fetch_oauth_credentials,
)
from shared.tools.google.gmail import GmailCreateDraftTool, GmailReadTool


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_schema_registry(reply_schema: JsonSchema) -> SchemaRegistry:
    registry = SchemaRegistry(
        {
            "schemas.gmail.reply@v1": reply_schema,
        }
    )
    return registry


def build_tool_registry(service: GmailDemoService) -> ToolRegistry:
    read_tool = GmailReadTool()
    draft_tool = GmailCreateDraftTool()

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
    registry.register(
        ToolDefinition(
            name="gmail_create_draft",
            version="v1",
            input_schema=draft_tool.get_parameters_schema(),
            output_schema=draft_tool.get_output_schema(),
            handler=lambda params, config: service.create_draft(params),
        )
    )
    return registry


def build_model_registry(reply_schema: JsonSchema) -> ModelRegistry:
    """
    Register a ChatOpenAI-backed handler that produces structured replies.
    """

    class ReplyPayload(BaseModel):
        to: List[str] = Field(..., description="List of recipient email addresses")
        subject: str = Field(..., description="Subject line for the reply")
        body_text: str = Field(..., description="Plain-text body for the reply")

    try:
        llm = ChatOpenAI(
            model='gpt-5-nano',
            temperature=0.2,
        )
    except Exception as exc:  # pragma: no cover - depends on env configuration
        raise RuntimeError(
            "ChatOpenAI initialization failed. Ensure OPENAI_API_KEY (and optional OPENAI_MODEL) "
            "are set in the environment."
        ) from exc
    structured_llm = llm.with_structured_output(ReplyPayload)

    def llm_handler(invocation: Dict[str, Any], schema: JsonSchema) -> Dict[str, Any]:
        email = invocation.get("inputs", {}).get("email", {})
        rendered_prompt = (
            f"{invocation['prompt']}\n\n"
            "Email content (JSON):\n"
            f"{json.dumps(email, indent=2)}\n\n"
            "Return the reply as structured JSON with keys to, subject, body_text."
        )
        reply = structured_llm.invoke(rendered_prompt)
        return reply.model_dump()

    registry = ModelRegistry()
    registry.register(
        ModelDefinition(
            model_id="gmail-demo-llm",
            json_handler=llm_handler,
        )
    )
    return registry


def build_workflow_spec(reply_schema: JsonSchema, read_schema: JsonSchema, draft_schema: JsonSchema) -> Dict[str, Any]:
    return {
        "version": "1",
        "inputs": {
            "user_id": {"type": "integer", "description": "Owner of the Gmail OAuth credentials", "required": True},
        },
        "nodes": [
            {
                "id": "fetch_email",
                "type": "tool",
                "tool": "gmail_read_emails",
                "in": {
                    "user_id": "${inputs.user_id}",
                    "max_results": 1,
                    "label_ids": ["INBOX"],
                    "include_body": True,
                },
                "out": "emails",
                "expect_output": {
                    "mode": "json",
                    "schema": {"json_schema": read_schema},
                },
            },
            {
                "id": "draft_reply",
                "type": "llm",
                "model": "gmail-demo-llm",
                "prompt": (
                    "You are a helpful assistant that writes short, friendly replies.\n"
                    "Original email subject: ${emails[0].subject}\n"
                    "Original body:\n${emails[0].body}\n\n"
                    "Respond succinctly and confirm next steps."
                ),
                "in": {"email": "${emails[0]}"},
                "out": "reply_payload",
                "output": {
                    "mode": "json",
                    "schema": {"json_schema": reply_schema},
                },
            },
            {
                "id": "save_draft",
                "type": "tool",
                "tool": "gmail_create_draft",
                "in": {
                    "user_id": "${inputs.user_id}",
                    "to": "${reply_payload.to}",
                    "subject": "${reply_payload.subject}",
                    "body_text": "${reply_payload.body_text}",
                },
                "out": "draft_info",
                "expect_output": {
                    "mode": "json",
                    "schema": {"json_schema": draft_schema},
                },
            },
        ],
        "output": "${draft_info}",
    }


def main() -> None:
    user_id = 1
    credentials = asyncio.run(fetch_oauth_credentials(user_id))
    service = GmailDemoService(credentials)

    reply_schema: JsonSchema = {
        "type": "object",
        "properties": {
            "to": {"type": "array", "items": {"type": "string"}},
            "subject": {"type": "string"},
            "body_text": {"type": "string"},
        },
        "required": ["to", "subject", "body_text"],
        "additionalProperties": False,
    }

    read_schema = service.read_tool.get_output_schema()
    draft_schema = service.create_draft_tool.get_output_schema()

    schema_registry = build_schema_registry(reply_schema)
    tool_registry = build_tool_registry(service)
    model_registry = build_model_registry(reply_schema)

    workflow_spec = build_workflow_spec(reply_schema, read_schema, draft_schema)
    context = CompilerContext(
        schema_registry=schema_registry,
        tool_registry=tool_registry,
        model_registry=model_registry,
    )

    compiled = compile_workflow(workflow_spec, context)
    result = compiled.invoke(inputs={"user_id": user_id})

    print("Workflow execution complete. Draft payload:")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


