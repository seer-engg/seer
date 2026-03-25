"""
Eval tests for embedding-based tool and trigger retrieval.

Design principles:
1. Each query is a realistic workflow scenario from the product's discovery UI
2. Every scenario asserts BOTH tool and trigger retrieval
3. Positive recall (any-of) + negative exclusion (wrong-domain)
4. Tools use top-10, triggers use top-5 (realistic window sizes)
5. Never weaken expectations to make tests pass — failures are signal

Requires OPENAI_API_KEY. Run with:
    uv run pytest tests/eval/test_retrieval_eval.py -s --tb=short
"""
from __future__ import annotations

import os

import pytest

from seer.tools.discovery_shared import (
    _embedding_index,
    async_search_tools_intent,
    async_search_triggers_intent,
)

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for embedding eval tests",
    ),
]


# ---------------------------------------------------------------------------
# Workflow scenarios: realistic queries from SuggestedPrompts.tsx
#
# Each scenario: (query, tool_expected_any_of, tool_must_not,
#                  trigger_expected_any_of, trigger_must_not)
# ---------------------------------------------------------------------------

WORKFLOW_SCENARIOS = [
    # 1. Lead outreach agent
    (
        "Build an outreach agent that finds new leads in a Google Sheet, "
        "enriches them with LinkedIn data, and sends personalized cold emails via Gmail.",
        ["google_sheets_read", "linkedin_get_profile", "gmail_send_email", "gmail_create_draft"],
        ["oura_get_sleep", "discord_send_channel_message"],
        ["schedule.cron", "webhook.generic", "form.hosted"],
        [],
    ),
    # 2. Email responder
    (
        "Create an automated email responder that reads incoming Gmail messages, "
        "looks up answers in my knowledge base, and drafts helpful replies.",
        ["gmail_read_emails", "gmail_get_message", "kb_query", "gmail_create_draft"],
        ["discord_send_channel_message", "linkedin_create_post"],
        ["poll.gmail.email_received"],
        [],
    ),
    # 3. Meeting follow-up
    (
        "Build a workflow that watches my Google Calendar for completed meetings, "
        "generates a summary in Google Docs, and emails follow-up action items to attendees.",
        ["google_calendar_list_events", "google_calendar_get_event",
         "google_docs_create", "google_docs_write", "gmail_send_email"],
        ["discord_send_channel_message", "youtube_upload_video"],
        ["poll.google_calendar.event_changed", "schedule.cron"],
        [],
    ),
    # 4. Content writer
    (
        "Create a content pipeline that researches trending topics on the web, "
        "writes LinkedIn posts on a schedule, and queues them for publishing.",
        ["web_search", "linkedin_create_post"],
        ["supabase_table_delete", "oura_get_stress"],
        ["schedule.cron"],
        [],
    ),
    # 5. CRM updater
    (
        "Build a workflow that captures form submissions, creates or updates entries "
        "in a Notion CRM database, and sends a confirmation email via Gmail.",
        ["notion_create_database_entry", "notion_create_page",
         "notion_query_database", "gmail_send_email"],
        ["youtube_upload_video", "oura_get_sleep"],
        ["form.hosted", "webhook.generic"],
        [],
    ),
    # 6. Slack alert bot
    (
        "Create a bot that monitors a Gmail inbox for important emails matching "
        "certain criteria and sends instant alerts to a Slack channel.",
        ["gmail_read_emails", "gmail_get_message", "slack_send_channel_message"],
        ["linkedin_create_post", "youtube_upload_video"],
        ["poll.gmail.email_received", "schedule.cron"],
        [],
    ),
]

TOOL_TOP_K = 10
TRIGGER_TOP_K = 5


@pytest.fixture(autouse=True)
def _reset_index():
    _embedding_index.invalidate()
    yield
    _embedding_index.invalidate()


async def _run_examples(search_fn, examples, key_field):
    """Run retrieval examples. Returns (failures, total_assertions)."""
    failures = []
    total = 0
    for item in examples:
        query, top_k, expected_any_of, must_not_appear = item
        results = await search_fn(query, top_k=top_k)
        result_keys = [r[key_field] for r in results[:top_k]]

        # Positive: at least one expected tool must appear
        total += 1
        if not any(e in result_keys for e in expected_any_of):
            failures.append(
                f"  MISS query={query!r} top-{top_k}: "
                f"wanted any of {expected_any_of}, got {result_keys}"
            )

        # Negative: wrong-domain tools must NOT appear
        for banned in must_not_appear:
            total += 1
            if banned in result_keys:
                failures.append(
                    f"  LEAK query={query!r} top-{top_k}: "
                    f"{banned!r} should not appear, got {result_keys}"
                )

    return failures, total


def _tool_examples():
    """Extract tool assertions from workflow scenarios."""
    return [
        (s[0], TOOL_TOP_K, s[1], s[2])
        for s in WORKFLOW_SCENARIOS
    ]


def _trigger_examples():
    """Extract trigger assertions from workflow scenarios."""
    return [
        (s[0], TRIGGER_TOP_K, s[3], s[4])
        for s in WORKFLOW_SCENARIOS
    ]


@pytest.mark.asyncio
async def test_tool_retrieval():
    """Tool search: positive recall + negative exclusion."""
    failures, total = await _run_examples(
        async_search_tools_intent, _tool_examples(), "name"
    )
    pct = (total - len(failures)) / total * 100 if total else 0
    summary = f"Tool retrieval: {total - len(failures)}/{total} ({pct:.0f}%)"
    print(f"\n{summary}")
    assert not failures, f"{summary}\n" + "\n".join(failures)


@pytest.mark.asyncio
async def test_trigger_retrieval():
    """Trigger search: positive recall + negative exclusion."""
    failures, total = await _run_examples(
        async_search_triggers_intent, _trigger_examples(), "key"
    )
    pct = (total - len(failures)) / total * 100 if total else 0
    summary = f"Trigger retrieval: {total - len(failures)}/{total} ({pct:.0f}%)"
    print(f"\n{summary}")
    assert not failures, f"{summary}\n" + "\n".join(failures)


@pytest.mark.asyncio
async def test_overall_pass_rate():
    """Combined pass rate across all assertions must be 100%."""
    tool_f, tool_t = await _run_examples(
        async_search_tools_intent, _tool_examples(), "name"
    )
    trig_f, trig_t = await _run_examples(
        async_search_triggers_intent, _trigger_examples(), "key"
    )
    total = tool_t + trig_t
    failed = len(tool_f) + len(trig_f)
    pct = (total - failed) / total * 100 if total else 0
    print(f"\nOverall: {total - failed}/{total} ({pct:.0f}%)")

    all_failures = tool_f + trig_f
    assert not all_failures, (
        f"Overall {total - failed}/{total} ({pct:.0f}%):\n" + "\n".join(all_failures)
    )
