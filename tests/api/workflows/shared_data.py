from __future__ import annotations

import logging
from typing import Any, Dict

from shared.tools.google.gmail import GmailReadTool


tool = GmailReadTool()
TOOL_NAME = tool.name
EMAIL_SCHEMA = tool.get_output_schema()
MODEL_ID = "gpt-5-nano"

TEST_WORKFLOW_SPEC_DATA: Dict[str, Any] = {
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
                "user_id": "${inputs.user_id}",
                "max_results": 3,
                "label_ids": ["INBOX"],
                "include_body": True,
            },
            "out": "emails",
            "expect_output": {
                "mode": "json",
                "schema": {"json_schema": EMAIL_SCHEMA},
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

TEST_USER_ID: int = 1
TEST_SCHEMA_ID = "schemas.IssueSummary@v1"
TEST_SCHEMA_DEFINITION: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["title", "summary"],
}


def configure_workflow_test_logging() -> None:
    logging.getLogger("api.workflows.services").setLevel(logging.DEBUG)
    logging.getLogger("workflow_compiler.runtime").setLevel(logging.DEBUG)

