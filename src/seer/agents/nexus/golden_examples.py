"""
Golden examples for workflow retrieval eval and agent few-shot prompting.

Each example is a (prompt, expected_tools, expected_triggers, spec) tuple that:
1. Tests retrieval recall — search_tools/search_triggers must find the right results
2. Provides few-shot examples — the agent can reference these for correct specs
3. Seeds frontend templates — suggested prompts for the chat UI
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class GoldenExample:
    """A validated prompt → tools → triggers → workflow spec mapping."""

    prompt: str
    expected_tools: List[str]
    expected_triggers: List[str]
    spec: Dict[str, Any]
    tags: List[str] = field(default_factory=list)


GOLDEN_EXAMPLES: List[GoldenExample] = [
    # 1. Cron → HTTP fetch → for_each → HITL review
    GoldenExample(
        prompt="Every morning at 9am, fetch top HN stories via API and let me review each one before posting to Slack",
        expected_tools=["http_request", "slack_send_channel_message"],
        expected_triggers=["schedule.cron"],
        tags=["schedule", "http", "slack", "hitl", "for_each"],
        spec={
            "version": "2",
            "triggers": [
                {
                    "id": "cron_1",
                    "key": "schedule.cron",
                    "mode": "polling",
                    "provider_config": {"cron_expression": "0 9 * * *", "timezone": "America/Los_Angeles"},
                    "event_schema": {},
                    "meta": {},
                    "filters": {},
                    "ui_meta": {},
                }
            ],
            "nodes": [
                {
                    "id": "fetch_hn",
                    "type": "tool",
                    "tool": "http_request",
                    "inputs": {
                        "method": "GET",
                        "url": "https://hacker-news.firebaseio.com/v0/topstories.json",
                    },
                },
                {
                    "id": "loop_stories",
                    "type": "for_each",
                    "items": "${fetch_hn.body}",
                },
                {
                    "id": "fetch_story",
                    "type": "tool",
                    "tool": "http_request",
                    "inputs": {
                        "method": "GET",
                        "url": "https://hacker-news.firebaseio.com/v0/item/${item}.json",
                    },
                },
                {
                    "id": "review",
                    "type": "hitl",
                    "title": "Review HN Story",
                    "description": "Review story before posting to Slack",
                    "display": [
                        {"label": "Title", "value": "${fetch_story.body.title}"},
                        {"label": "URL", "value": "${fetch_story.body.url}"},
                    ],
                    "inputs": [
                        {
                            "id": "approved",
                            "question": "Post this story to Slack?",
                            "input_type": "boolean",
                        }
                    ],
                },
                {
                    "id": "check_approved",
                    "type": "if",
                    "condition": "${review.approved}",
                },
                {
                    "id": "post_slack",
                    "type": "tool",
                    "tool": "slack_send_channel_message",
                    "inputs": {
                        "channel": "#news",
                        "message": "${fetch_story.body.title}\n${fetch_story.body.url}",
                    },
                },
            ],
            "edges": [
                {"source": "cron_1", "target": "fetch_hn", "type": "trigger"},
                {"source": "fetch_hn", "target": "loop_stories", "type": "default"},
                {"source": "loop_stories", "target": "fetch_story", "type": "loop_body"},
                {"source": "fetch_story", "target": "review", "type": "default"},
                {"source": "review", "target": "check_approved", "type": "default"},
                {"source": "check_approved", "target": "post_slack", "type": "conditional_true"},
                {"source": "loop_stories", "target": "loop_stories", "type": "loop_exit"},
            ],
        },
    ),

    # 2. Supabase row insert → Gmail welcome email
    GoldenExample(
        prompt="When a new user signs up in Supabase, send them a welcome email via Gmail",
        expected_tools=["gmail_send_email"],
        expected_triggers=["webhook.supabase.db_changes"],
        tags=["supabase", "gmail", "welcome", "onboarding"],
        spec={
            "version": "2",
            "triggers": [
                {
                    "id": "supabase_signup",
                    "key": "webhook.supabase.db_changes",
                    "mode": "webhook",
                    "provider_config": {"table": "users", "event": "INSERT"},
                    "event_schema": {
                        "type": "object",
                        "properties": {
                            "record": {
                                "type": "object",
                                "properties": {
                                    "email": {"type": "string"},
                                    "name": {"type": "string"},
                                },
                            }
                        },
                    },
                    "meta": {},
                    "filters": {},
                    "ui_meta": {},
                }
            ],
            "nodes": [
                {
                    "id": "send_welcome",
                    "type": "tool",
                    "tool": "gmail_send_email",
                    "inputs": {
                        "to": "${supabase_signup.event.record.email}",
                        "subject": "Welcome to our platform!",
                        "body": "Hi ${supabase_signup.event.record.name},\n\nWelcome aboard! We're glad to have you.",
                    },
                }
            ],
            "edges": [
                {"source": "supabase_signup", "target": "send_welcome", "type": "trigger"},
            ],
        },
    ),

    # 3. Schedule → Slack daily summary
    GoldenExample(
        prompt="Send a daily Slack summary of today's Google Calendar events at 8am",
        expected_tools=["google_calendar_list_events", "slack_send_channel_message"],
        expected_triggers=["schedule.cron"],
        tags=["schedule", "google_calendar", "slack", "daily"],
        spec={
            "version": "2",
            "triggers": [
                {
                    "id": "daily_8am",
                    "key": "schedule.cron",
                    "mode": "polling",
                    "provider_config": {"cron_expression": "0 8 * * *", "timezone": "America/Los_Angeles"},
                    "event_schema": {},
                    "meta": {},
                    "filters": {},
                    "ui_meta": {},
                }
            ],
            "nodes": [
                {
                    "id": "get_events",
                    "type": "tool",
                    "tool": "google_calendar_list_events",
                    "inputs": {
                        "time_min": "${now()}",
                        "time_max": "${end_of_day()}",
                    },
                },
                {
                    "id": "post_summary",
                    "type": "tool",
                    "tool": "slack_send_channel_message",
                    "inputs": {
                        "channel": "#daily-standup",
                        "message": "Today's events:\n${get_events.output}",
                    },
                },
            ],
            "edges": [
                {"source": "daily_8am", "target": "get_events", "type": "trigger"},
                {"source": "get_events", "target": "post_summary", "type": "default"},
            ],
        },
    ),

    # 4. Gmail new email → if subject match → Slack notify
    GoldenExample(
        prompt="When I get a Gmail with 'URGENT' in the subject, send a Slack DM to me",
        expected_tools=["slack_send_direct_message"],
        expected_triggers=["poll.gmail.email_received"],
        tags=["gmail", "slack", "filter", "notification"],
        spec={
            "version": "2",
            "triggers": [
                {
                    "id": "gmail_new",
                    "key": "poll.gmail.email_received",
                    "mode": "polling",
                    "provider_config": {},
                    "event_schema": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "from": {"type": "string"},
                            "snippet": {"type": "string"},
                        },
                    },
                    "meta": {},
                    "filters": {},
                    "ui_meta": {},
                }
            ],
            "nodes": [
                {
                    "id": "check_urgent",
                    "type": "if",
                    "condition": "${'URGENT' in gmail_new.event.subject}",
                },
                {
                    "id": "notify_slack",
                    "type": "tool",
                    "tool": "slack_send_direct_message",
                    "inputs": {
                        "user": "me",
                        "message": "Urgent email from ${gmail_new.event.from}: ${gmail_new.event.subject}",
                    },
                },
            ],
            "edges": [
                {"source": "gmail_new", "target": "check_urgent", "type": "trigger"},
                {"source": "check_urgent", "target": "notify_slack", "type": "conditional_true"},
            ],
        },
    ),

    # 5. Webhook → HTTP API call → response check
    GoldenExample(
        prompt="When a webhook fires, call an external API and post the result to Slack if successful",
        expected_tools=["http_request", "slack_send_channel_message"],
        expected_triggers=["webhook.generic"],
        tags=["webhook", "http", "slack", "conditional"],
        spec={
            "version": "2",
            "triggers": [
                {
                    "id": "webhook_in",
                    "key": "webhook.generic",
                    "mode": "webhook",
                    "provider_config": {},
                    "event_schema": {"type": "object"},
                    "meta": {},
                    "filters": {},
                    "ui_meta": {},
                }
            ],
            "nodes": [
                {
                    "id": "call_api",
                    "type": "tool",
                    "tool": "http_request",
                    "inputs": {
                        "method": "POST",
                        "url": "https://api.example.com/process",
                        "body": "${webhook_in.event}",
                    },
                },
                {
                    "id": "check_status",
                    "type": "if",
                    "condition": "${call_api.status_code == 200}",
                },
                {
                    "id": "post_result",
                    "type": "tool",
                    "tool": "slack_send_channel_message",
                    "inputs": {
                        "channel": "#api-results",
                        "message": "API response: ${call_api.body}",
                    },
                },
            ],
            "edges": [
                {"source": "webhook_in", "target": "call_api", "type": "trigger"},
                {"source": "call_api", "target": "check_status", "type": "default"},
                {"source": "check_status", "target": "post_result", "type": "conditional_true"},
            ],
        },
    ),

    # 6. Supabase insert → loop rows → Notion page per row
    GoldenExample(
        prompt="Query new orders from Supabase and create a Notion page for each one",
        expected_tools=["supabase_table_query", "notion_create_page"],
        expected_triggers=["schedule.cron"],
        tags=["supabase", "notion", "for_each", "sync"],
        spec={
            "version": "2",
            "triggers": [
                {
                    "id": "hourly",
                    "key": "schedule.cron",
                    "mode": "polling",
                    "provider_config": {"cron_expression": "0 * * * *", "timezone": "America/Los_Angeles"},
                    "event_schema": {},
                    "meta": {},
                    "filters": {},
                    "ui_meta": {},
                }
            ],
            "nodes": [
                {
                    "id": "query_orders",
                    "type": "tool",
                    "tool": "supabase_table_query",
                    "inputs": {
                        "table": "orders",
                        "filters": {"status": "new"},
                    },
                },
                {
                    "id": "loop_orders",
                    "type": "for_each",
                    "items": "${query_orders.output}",
                },
                {
                    "id": "create_notion_page",
                    "type": "tool",
                    "tool": "notion_create_page",
                    "inputs": {
                        "title": "Order #${item.id}",
                        "content": "Customer: ${item.customer}\nTotal: ${item.total}",
                    },
                },
            ],
            "edges": [
                {"source": "hourly", "target": "query_orders", "type": "trigger"},
                {"source": "query_orders", "target": "loop_orders", "type": "default"},
                {"source": "loop_orders", "target": "create_notion_page", "type": "loop_body"},
                {"source": "loop_orders", "target": "loop_orders", "type": "loop_exit"},
            ],
        },
    ),
]


def get_examples_by_tag(tag: str) -> List[GoldenExample]:
    """Filter golden examples by tag."""
    return [e for e in GOLDEN_EXAMPLES if tag in e.tags]


def get_example_prompts() -> List[str]:
    """Get all golden example prompts (for frontend template suggestions)."""
    return [e.prompt for e in GOLDEN_EXAMPLES]
