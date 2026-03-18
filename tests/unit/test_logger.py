"""Unit tests for Slack log formatting and delivery."""

from __future__ import annotations

import logging
import sys

import pytest

from seer.logger import SlackHandler


def _make_record(*, message: str = "Something failed", level: int = logging.ERROR, extras: dict | None = None, exc_info=None) -> logging.LogRecord:
    """Create a log record with optional extra context."""
    record_data = {
        "name": "seer.test",
        "levelno": level,
        "levelname": logging.getLevelName(level),
        "pathname": __file__,
        "filename": "test_logger.py",
        "module": "test_logger",
        "lineno": 42,
        "msg": message,
        "args": (),
        "exc_info": exc_info,
    }
    if extras:
        record_data.update(extras)
    return logging.makeLogRecord(record_data)


class _MockResponse:
    """Minimal HTTP response stub for Slack API calls."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _MockAsyncClient:
    """Capture posted Slack payloads."""

    def __init__(self):
        self.posts: list[dict] = []

    async def post(self, url: str, json: dict, headers: dict) -> _MockResponse:  # pylint: disable=redefined-builtin  # Reason: Match httpx API
        self.posts.append({"url": url, "json": json, "headers": headers})
        if "thread_ts" in json:
            return _MockResponse({"ok": True})
        return _MockResponse({"ok": True, "ts": "1234.5678"})


@pytest.mark.unit
def test_format_slack_main_message_excludes_extra_context():
    """Extra fields should not be in the parent Slack message."""
    handler = SlackHandler(bot_token="token", channel_id="channel")
    record = _make_record(extras={"subscription_id": 123, "event_id": 456})

    message = handler._format_slack_main_message(record)
    thread_details = handler._format_slack_thread_details(record)

    assert "Extra Context:" not in message
    assert "subscription_id" not in message
    assert "(Additional details in thread)" in message
    assert thread_details is not None
    assert "Extra Context:" in thread_details
    assert "subscription_id: 123" in thread_details
    assert "event_id: 456" in thread_details


@pytest.mark.unit
@pytest.mark.asyncio
async def test_send_slack_notification_posts_extra_context_in_thread():
    """Extra fields should be posted as a thread reply even without an exception."""
    handler = SlackHandler(bot_token="token", channel_id="channel")
    handler._http_client = _MockAsyncClient()
    record = _make_record(extras={"subscription_id": 123})

    await handler._send_slack_notification(record)

    assert len(handler._http_client.posts) == 2

    main_post = handler._http_client.posts[0]["json"]
    thread_post = handler._http_client.posts[1]["json"]

    assert "thread_ts" not in main_post
    assert "Extra Context:" not in main_post["text"]
    assert thread_post["thread_ts"] == "1234.5678"
    assert "Extra Context:" in thread_post["text"]
    assert "subscription_id: 123" in thread_post["text"]


@pytest.mark.unit
def test_format_slack_thread_details_combines_extra_context_and_stack_trace():
    """Thread details should include both extras and a stack trace."""
    handler = SlackHandler(bot_token="token", channel_id="channel")

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = _make_record(extras={"subscription_id": 123}, exc_info=sys.exc_info())

    thread_details = handler._format_slack_thread_details(record)

    assert thread_details is not None
    assert "Extra Context:" in thread_details
    assert "subscription_id: 123" in thread_details
    assert "Stack Trace:" in thread_details
    assert "RuntimeError: boom" in thread_details
