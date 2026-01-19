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
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from tortoise import Tortoise

from seer.database.config import TORTOISE_ORM
from seer.database.models import User
from seer.tools.google.gmail import GmailReadTool
from seer.core.examples.gmail_common import GmailDemoService
from seer.core.registry.model_registry import ModelDefinition, ModelRegistry
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.runtime import WorkflowCompilerSingleton
from seer.core.schema.models import JsonSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_tortoise():
    await Tortoise.init(config=TORTOISE_ORM)


async def close_tortoise():
    await Tortoise.close_connections()

TOOL_NAME = "gmail_read_emails"
MODEL_ID = "gpt-5-nano"


def register_demo_components(service: GmailDemoService) -> None:
    compiler = WorkflowCompilerSingleton.instance()

    read_tool = GmailReadTool()
    tool_registry: ToolRegistry = compiler._tool_registry  # type: ignore[attr-defined]
    if not tool_registry.maybe_get(TOOL_NAME):
        tool_registry.register(
            ToolDefinition(
                name=TOOL_NAME,
                version="v1",
                input_schema=read_tool.get_parameters_schema(),
                output_schema=read_tool.get_output_schema(),
                handler=lambda params, config: service.read_emails(params),
            )
        )

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

    model_registry: ModelRegistry = compiler._model_registry  # type: ignore[attr-defined]
    if not model_registry.maybe_get(MODEL_ID):
        model_registry.register(
            ModelDefinition(
                model_id=MODEL_ID,
                text_handler=llm_handler,
            )
        )


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
                "tool": TOOL_NAME,
                "in": {
                    "user_id": "${trigger.data.user_id}",
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
                "model": MODEL_ID,
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


read_tool = GmailReadTool()
read_schema = read_tool.get_output_schema()


async def main() -> None:
    user_id = 1
    await init_tortoise()

    spec = build_workflow_spec(read_schema)
    compiler = WorkflowCompilerSingleton.instance()
    demo_user = await User.get(id=1)
    compiled = compiler.compile(demo_user, spec)
    summary = await compiled.ainvoke(trigger={"trigger_key": "manual.trigger", "data": {"user_id": user_id}})
    print("Inbox summary:\n", summary.get("inbox_summary"))
    await close_tortoise()


if __name__ == "__main__":
    asyncio.run(main())
