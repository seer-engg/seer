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
from typing import Any, Dict, List, Mapping, Optional, TypedDict

from tortoise import Tortoise

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from workflow_compiler import compile_workflow
from workflow_compiler.compiler.context import CompilerContext
from workflow_compiler.registry.model_registry import ModelDefinition, ModelRegistry
from workflow_compiler.registry.tool_registry import ToolDefinition, ToolRegistry
from workflow_compiler.schema.schema_registry import SchemaRegistry
from workflow_compiler.schema.models import JsonSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from shared.database.config import TORTOISE_ORM
except RuntimeError as exc:  # pragma: no cover - depends on env
    logger.warning("DATABASE_URL not configured (%s); DB lookups will be skipped", exc)
    TORTOISE_ORM = None

try:
    from shared.database.models_oauth import OAuthConnection
except RuntimeError as exc:  # pragma: no cover - depends on env
    logger.warning("Unable to import OAuthConnection (%s); DB lookups will be skipped", exc)
    OAuthConnection = None  # type: ignore[assignment]

from shared.tools.google.gmail import GmailCreateDraftTool, GmailReadTool


USE_REAL_GMAIL = os.getenv("USE_REAL_GMAIL", "0").lower() in {"1", "true", "yes"}


class GmailCredentials(TypedDict, total=False):
    access_token: str
    refresh_token: Optional[str]
    provider: str


async def fetch_oauth_credentials(user_id: int, provider: str = "google") -> Optional[GmailCredentials]:
    """
    Fetch OAuth credentials for the demo user. If the database or connection
    fails this returns None so the script can fall back to stubbed data.
    """

    if TORTOISE_ORM is None or OAuthConnection is None:
        logger.info("Skipping OAuth lookup because DATABASE_URL is not configured")
        return None

    try:  # pragma: no cover - requires DB connectivity
        await Tortoise.init(config=TORTOISE_ORM)
        conn = await OAuthConnection.filter(user_id=user_id, provider=provider).first()
        if conn:
            logger.info("Loaded OAuth credentials for user_id=%s provider=%s", user_id, provider)
            return GmailCredentials(
                access_token=conn.access_token_enc,
                refresh_token=conn.refresh_token_enc,
                provider=provider,
            )
    except Exception as exc:  # pragma: no cover - best-effort helper
        logger.warning("Unable to fetch OAuth credentials, continuing with stub data: %s", exc)
    return None


class GmailDemoService:
    """
    Lightweight adapter that either calls the real Gmail tools (when valid
    credentials + USE_REAL_GMAIL=1) or falls back to deterministic stub data.
    """

    def __init__(self, credentials: Optional[GmailCredentials]) -> None:
        self.credentials = credentials
        self.read_tool = GmailReadTool()
        self.create_draft_tool = GmailCreateDraftTool()

    def read_emails(self, params: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if USE_REAL_GMAIL and self.credentials:
            logger.info("Fetching Gmail messages via API")
            return asyncio.run(  # pragma: no cover - requires live credentials
                self.read_tool.execute(self.credentials["access_token"], dict(params))
            )

        logger.info("Using stub Gmail inbox data")
        max_results = int(params.get("max_results", 1))
        sample = {
            "id": "demo-message-1",
            "threadId": "demo-thread",
            "snippet": "Reminder about tomorrow's demo",
            "subject": "Demo tomorrow?",
            "from": "product@example.com",
            "to": "you@example.com",
            "date": "Fri, 13 Dec 2025 10:00:00 -0000",
            "labelIds": ["INBOX"],
            "body": (
                "Hey there,\n\nCan we confirm the talking points for tomorrow's demo?\n"
                "Let me know if you need anything else.\n\n- Product"
            ),
        }
        return [sample] * max(1, max_results)

    def create_draft(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        if USE_REAL_GMAIL and self.credentials:
            logger.info("Creating Gmail draft via API")
            payload = {
                "to": params.get("to", []),
                "subject": params.get("subject"),
                "body_text": params.get("body_text"),
                "body_html": params.get("body_html"),
            }
            return asyncio.run(  # pragma: no cover - requires live credentials
                self.create_draft_tool.execute(self.credentials["access_token"], payload)
            )

        logger.info("Stubbing Gmail draft creation")
        to_list = params.get("to") or []
        return {
            "id": "demo-draft-1",
            "message": {
                "id": "demo-message-2",
                "threadId": "demo-thread",
                "labelIds": ["DRAFT"],
                "snippet": (params.get("body_text") or "")[:140],
                "payload": {
                    "headers": [
                        {"name": "To", "value": ", ".join(to_list)},
                        {"name": "Subject", "value": params.get("subject", "")},
                    ],
                    "body": {"size": len(params.get("body_text", ""))},
                },
            },
        }


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

    def llm_handler(prompt: str, request: Dict[str, Any]) -> Dict[str, Any]:
        email = request.get("inputs", {}).get("email", {})
        rendered_prompt = (
            f"{prompt}\n\n"
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
            handler=llm_handler,
            supports_structured_output=True,
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


