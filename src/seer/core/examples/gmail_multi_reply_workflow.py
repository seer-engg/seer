"""
Example workflow that:

1. Fetches the latest three Gmail messages.
2. Uses an LLM to craft a personalized reply for each email.
3. Creates a Gmail draft for every generated reply.

Usage:
    OPENAI_API_KEY=... python -m workflow_compiler.examples.gmail_multi_reply_workflow

Set USE_REAL_GMAIL=1 (plus valid OAuth credentials for user 1) to hit the live Gmail
API. Otherwise the helper service falls back to deterministic stub data.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from seer.database.models import User
from seer.core.examples.gmail_common import (
    GmailDemoService,
    fetch_oauth_credentials,
)
from seer.core.examples.gmail_reply_workflow import (
    DRAFT_TOOL,
    MODEL_ID,
    READ_TOOL,
    register_demo_components,
)
from seer.core.runtime import WorkflowCompilerSingleton
from seer.core.schema.models import JsonSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_workflow_spec(
    reply_schema: JsonSchema,
    read_schema: JsonSchema,
    draft_schema: JsonSchema,
) -> Dict[str, Any]:
    """
    Construct a workflow specification that loops over fetched emails and drafts replies.
    """

    return {
        "version": "1",
        "inputs": {
            "user_id": {
                "type": "integer",
                "required": True,
                "description": "Owner of the Gmail mailbox",
            },
        },
        "nodes": [
            {
                "id": "fetch_emails",
                "type": "tool",
                "tool": READ_TOOL,
                "in": {
                    "user_id": "${trigger.data.user_id}",
                    "max_results": 3,
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
                "id": "reply_to_each_email",
                "type": "for_each",
                "items": "${emails}",
                "item_var": "email_item",
                "index_var": "email_index",
                "body": [
                    {
                        "id": "compose_reply",
                        "type": "llm",
                        "model": MODEL_ID,
                        "prompt": (
                            "You are a helpful assistant drafting a concise, friendly reply.\n"
                            "Keep the tone professional, propose next steps when relevant, and stay under 120 words.\n"
                            "Email #${email_index} subject: ${email_item.subject}\n"
                            "From: ${email_item.from}\n"
                            "Body:\n${email_item.body}\n"
                        ),
                        "in": {"email": "${email_item}"},
                        "out": "loop_reply_payload",
                        "output": {
                            "mode": "json",
                            "schema": {"json_schema": reply_schema},
                        },
                    },
                    {
                        "id": "create_reply_draft",
                        "type": "tool",
                        "tool": DRAFT_TOOL,
                        "in": {
                            "user_id": "${trigger.data.user_id}",
                            "to": "${loop_reply_payload.to}",
                            "subject": "${loop_reply_payload.subject}",
                            "body_text": "${loop_reply_payload.body_text}",
                            "thread_id": "${email_item.threadId}",
                        },
                        "out": "loop_draft_record",
                        "expect_output": {
                            "mode": "json",
                            "schema": {"json_schema": draft_schema},
                        },
                    },
                ],
                "out": "drafts",
                "output": {
                    "mode": "json",
                    "schema": {
                        "json_schema": {
                            "type": "array",
                            "items": draft_schema,
                        }
                    },
                },
            },
        ],
        "output": {
            "drafts": "${drafts}",
        },
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

    register_demo_components(service)
    compiler = WorkflowCompilerSingleton.instance()
    demo_user = User(id=0, user_id="demo-gmail-multi-reply")

    workflow_spec = build_workflow_spec(reply_schema, read_schema, draft_schema)
    compiled = compiler.compile(demo_user, workflow_spec)
    result = compiled.invoke(trigger={"trigger_key": "manual.trigger", "data": {"user_id": user_id}})

    drafts = result.get("drafts", [])
    print(f"Created {len(drafts)} draft(s).")
    for idx, draft in enumerate(drafts, start=1):
        snippet = draft.get("message", {}).get("snippet", "")
        print(f"[Draft #{idx}] snippet={snippet!r}")


if __name__ == "__main__":
    main()
