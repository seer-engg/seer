# pylint: disable=too-many-lines
# Reason: Comprehensive E2E test for email triage workflow with detailed mock implementations
"""
E2E Integration Test for Email Triage Workflow.

This test file investigates the runtime error:
    "Reference root 'fetch_product_spec' does not match the active trigger ID 'incoming_email'"

The test uses the exact workflow spec from the error report to trace through
the expression evaluator and understand why tool node outputs might not be
available in state when referenced by subsequent nodes.

Key investigation points:
1. Are both tool nodes (fetch_product_spec, fetch_full_email) executing?
2. Are their outputs being added to state correctly?
3. Is the LLM node receiving the tool outputs?
4. Does the conditional branch work?
"""
from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from seer.core.registry.model_registry import ModelDefinition
from seer.core.registry.tool_registry import ToolDefinition

from .conftest import compile_workflow


# =============================================================================
# EMAIL TRIAGE WORKFLOW SPECIFICATION
# =============================================================================
# This is the exact workflow spec from the error report that produces:
# "Reference root 'fetch_product_spec' does not match the active trigger ID 'incoming_email'"


EMAIL_TRIAGE_WORKFLOW_SPEC: Dict[str, Any] = {
    "version": "2",
    "triggers": [
        {
            "id": "incoming_email",
            "key": "poll.gmail.email_received",
            "mode": "poll",
            "event_schema": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "thread_id": {"type": "string"},
                    "internal_date_ms": {"type": "integer"},
                    "subject": {"type": "string"},
                    "snippet": {"type": "string"},
                    "from": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                        },
                    },
                    "to": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                            },
                        },
                    },
                    "labels": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    ],
    "nodes": [
        {
            "id": "fetch_product_spec",
            "type": "tool",
            "tool": "google_docs_read",
            "inputs": {
                "document_id": "1ABC123_test_document_id",  # Static document ID for testing
            },
        },
        {
            "id": "fetch_full_email",
            "type": "tool",
            "tool": "gmail_get_message",
            "inputs": {
                "message_id": "${incoming_email.message_id}",
                "format": "full",
            },
        },
        {
            "id": "analyze_and_generate_response",
            "type": "agent",
            "inputs": {
                "model": "openai/gpt-oss-120b",
                "prompt": """You are an AI assistant helping to triage incoming emails.

Product Specification:
${fetch_product_spec.body}

Email Details:
Subject: ${fetch_full_email.subject}
From: ${fetch_full_email.from}
Body:
${fetch_full_email.body}

Your task:
1. Analyze if this email requires a response
2. If yes, classify it and generate an appropriate reply
3. Be professional and helpful""",
            },
            "outputs": {
                "mode": "json",
                "schema": {
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "classification": {
                                "type": "string",
                                "enum": ["inquiry", "complaint", "feedback", "spam", "other"],
                            },
                            "should_reply": {"type": "boolean"},
                            "reasoning": {"type": "string"},
                            "sender_email": {"type": "string"},
                            "sender_name": {"type": "string"},
                            "reply_to_email": {"type": "string"},
                            "reply_subject": {"type": "string"},
                            "reply_body": {"type": "string"},
                        },
                        "required": [
                            "classification",
                            "should_reply",
                            "reasoning",
                            "sender_email",
                            "sender_name",
                            "reply_to_email",
                            "reply_subject",
                            "reply_body",
                        ],
                    }
                },
            },
        },
        {
            "id": "check_should_reply",
            "type": "if",
            "condition": "${analyze_and_generate_response.should_reply}",
        },
        {
            "id": "send_reply",
            "type": "tool",
            "tool": "gmail_send_email",
            "inputs": {
                "to": "${analyze_and_generate_response.reply_to_email}",
                "subject": "${analyze_and_generate_response.reply_subject}",
                "body": "${analyze_and_generate_response.reply_body}",
                "thread_id": "${incoming_email.thread_id}",
            },
        },
    ],
    "edges": [
        {"source": "incoming_email", "target": "fetch_product_spec", "type": "trigger"},
        {"source": "incoming_email", "target": "fetch_full_email", "type": "trigger"},
        {"source": "fetch_product_spec", "target": "analyze_and_generate_response", "type": "default"},
        {"source": "fetch_full_email", "target": "analyze_and_generate_response", "type": "default"},
        {"source": "analyze_and_generate_response", "target": "check_should_reply", "type": "default"},
        {"source": "check_should_reply", "target": "send_reply", "type": "conditional_true"},
    ],
}


# =============================================================================
# MOCK TOOL FACTORIES
# =============================================================================


def create_google_docs_read_tool(call_tracker: List[str]) -> ToolDefinition:
    """
    Create a mock google_docs_read tool that returns a product specification.

    This simulates fetching a Google Doc with product documentation.

    Args:
        call_tracker: List to append tool invocation records for verification.

    Returns:
        ToolDefinition for the mock google_docs_read tool.
    """

    def handler(inputs: Dict[str, Any], _config: Any, _context: Any) -> Dict[str, Any]:
        call_tracker.append(f"google_docs_read:{inputs.get('document_id', 'unknown')}")
        return {
            "documentId": inputs.get("document_id"),
            "title": "Product Specification",
            "body": (
                "Our product is an AI-powered email assistant. "
                "Key features: automatic triage, smart replies, priority detection. "
                "Support contact: support@company.com"
            ),
        }

    async def async_handler(inputs: Dict[str, Any], _config: Any, _context: Any) -> Dict[str, Any]:
        call_tracker.append(f"google_docs_read:{inputs.get('document_id', 'unknown')}")
        return {
            "documentId": inputs.get("document_id"),
            "title": "Product Specification",
            "body": (
                "Our product is an AI-powered email assistant. "
                "Key features: automatic triage, smart replies, priority detection. "
                "Support contact: support@company.com"
            ),
        }

    return ToolDefinition(
        name="google_docs_read",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "The ID of the Google Doc to read"},
            },
            "required": ["document_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "documentId": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
        },
        handler=handler,
        async_handler=async_handler,
    )


def create_gmail_get_message_tool(call_tracker: List[str]) -> ToolDefinition:
    """
    Create a mock gmail_get_message tool that returns email details.

    This simulates fetching full email content from Gmail API.

    Args:
        call_tracker: List to append tool invocation records for verification.

    Returns:
        ToolDefinition for the mock gmail_get_message tool.
    """

    def handler(inputs: Dict[str, Any], _config: Any, _context: Any) -> Dict[str, Any]:
        call_tracker.append(f"gmail_get_message:{inputs.get('message_id', 'unknown')}")
        return {
            "id": inputs.get("message_id"),
            "threadId": "thread_test_456",
            "subject": "Question about your product",
            "from": "testuser@example.com",
            "body": (
                "Hello,\n\n"
                "I've been using your email assistant and have a question about "
                "the automatic triage feature. How does it determine priority?\n\n"
                "Best regards,\n"
                "Test User"
            ),
        }

    async def async_handler(inputs: Dict[str, Any], _config: Any, _context: Any) -> Dict[str, Any]:
        call_tracker.append(f"gmail_get_message:{inputs.get('message_id', 'unknown')}")
        return {
            "id": inputs.get("message_id"),
            "threadId": "thread_test_456",
            "subject": "Question about your product",
            "from": "testuser@example.com",
            "body": (
                "Hello,\n\n"
                "I've been using your email assistant and have a question about "
                "the automatic triage feature. How does it determine priority?\n\n"
                "Best regards,\n"
                "Test User"
            ),
        }

    return ToolDefinition(
        name="gmail_get_message",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "description": "The ID of the email message"},
                "format": {"type": "string", "enum": ["minimal", "full", "raw", "metadata"]},
            },
            "required": ["message_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "threadId": {"type": "string"},
                "subject": {"type": "string"},
                "from": {"type": "string"},
                "body": {"type": "string"},
            },
        },
        handler=handler,
        async_handler=async_handler,
    )


def create_gmail_send_email_tool(call_tracker: List[str]) -> ToolDefinition:
    """
    Create a mock gmail_send_email tool that simulates sending an email.

    Args:
        call_tracker: List to append tool invocation records for verification.

    Returns:
        ToolDefinition for the mock gmail_send_email tool.
    """

    def handler(inputs: Dict[str, Any], _config: Any, _context: Any) -> Dict[str, Any]:
        call_tracker.append(f"gmail_send_email:to={inputs.get('to', 'unknown')}")
        return {
            "message_id": "sent_msg_123",
            "thread_id": inputs.get("thread_id", "new_thread"),
            "status": "sent",
        }

    async def async_handler(inputs: Dict[str, Any], _config: Any, _context: Any) -> Dict[str, Any]:
        call_tracker.append(f"gmail_send_email:to={inputs.get('to', 'unknown')}")
        return {
            "message_id": "sent_msg_123",
            "thread_id": inputs.get("thread_id", "new_thread"),
            "status": "sent",
        }

    return ToolDefinition(
        name="gmail_send_email",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body content"},
                "thread_id": {"type": "string", "description": "Thread ID to reply to"},
            },
            "required": ["to", "subject", "body"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "message_id": {"type": "string"},
                "thread_id": {"type": "string"},
                "status": {"type": "string"},
            },
        },
        handler=handler,
        async_handler=async_handler,
    )


# =============================================================================
# MOCK LLM HANDLER FACTORY
# =============================================================================


def create_mock_email_classification_handler():
    """
    Create a mock LLM handler that returns email classification results.

    This returns a response in the exact schema expected by the workflow's
    analyze_and_generate_response node.

    Returns:
        Callable that returns (result, usage_metadata) tuple.
    """

    def handler(_invocation: Dict[str, Any], _schema: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        # Return classification with should_reply=True to test the full flow
        result = {
            "classification": "inquiry",
            "should_reply": True,
            "reasoning": "This is a product question that deserves a helpful response.",
            "sender_email": "testuser@example.com",
            "sender_name": "Test User",
            "reply_to_email": "testuser@example.com",
            "reply_subject": "Re: Question about your product",
            "reply_body": (
                "Hello Test User,\n\n"
                "Thank you for your question about our automatic triage feature! "
                "The system uses machine learning to analyze email content, sender history, "
                "and urgency indicators to determine priority.\n\n"
                "Let me know if you have any other questions.\n\n"
                "Best regards,\n"
                "AI Assistant"
            ),
        }
        return result, {}

    return handler


def create_mock_no_reply_handler():
    """
    Create a mock LLM handler that returns should_reply=False.

    Used to test the conditional branch when no reply is needed.

    Returns:
        Callable that returns (result, usage_metadata) tuple.
    """

    def handler(_invocation: Dict[str, Any], _schema: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        result = {
            "classification": "spam",
            "should_reply": False,
            "reasoning": "This appears to be spam and does not require a response.",
            "sender_email": "spammer@example.com",
            "sender_name": "Spammer",
            "reply_to_email": "spammer@example.com",
            "reply_subject": "",
            "reply_body": "",
        }
        return result, {}

    return handler


_INQUIRY_RESPONSE = {
    "classification": "inquiry",
    "should_reply": True,
    "reasoning": "This is a product question that deserves a helpful response.",
    "sender_email": "testuser@example.com",
    "sender_name": "Test User",
    "reply_to_email": "testuser@example.com",
    "reply_subject": "Re: Question about your product",
    "reply_body": (
        "Hello Test User,\n\n"
        "Thank you for your question about our automatic triage feature! "
        "The system uses machine learning to analyze email content, sender history, "
        "and urgency indicators to determine priority.\n\n"
        "Let me know if you have any other questions.\n\n"
        "Best regards,\n"
        "AI Assistant"
    ),
}

_NO_REPLY_RESPONSE = {
    "classification": "spam",
    "should_reply": False,
    "reasoning": "This appears to be spam and does not require a response.",
    "sender_email": "spammer@example.com",
    "sender_name": "Spammer",
    "reply_to_email": "spammer@example.com",
    "reply_subject": "",
    "reply_body": "",
}


def _create_inquiry_mock_agent() -> AsyncMock:
    """Create a mock agent that returns an inquiry classification response."""
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [AIMessage(content=json.dumps(_INQUIRY_RESPONSE))]
    }
    return mock_agent


def _create_no_reply_mock_agent() -> AsyncMock:
    """Create a mock agent that returns a no-reply (spam) classification response."""
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [AIMessage(content=json.dumps(_NO_REPLY_RESPONSE))]
    }
    return mock_agent


# =============================================================================
# E2E TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_email_triage_workflow_full_execution() -> None:
    """
    Test the email triage workflow executes completely with all nodes.

    This test traces through the exact workflow spec to investigate:
    - Whether both fetch_product_spec and fetch_full_email execute
    - Whether their outputs are in state for the LLM node
    - Whether the conditional branch executes correctly
    - Whether send_reply executes when should_reply=True

    If this test passes, the workflow spec is valid and production errors
    are likely due to OAuth/credentials or network issues.

    If this test fails with "Reference root 'fetch_product_spec' does not
    match the active trigger ID 'incoming_email'", there's a bug in:
    - Expression evaluation (the _resolve_root logic)
    - State propagation between nodes
    - Graph topology/execution order
    """
    call_tracker: List[str] = []

    # Create mock tools
    google_docs_tool = create_google_docs_read_tool(call_tracker)
    gmail_get_tool = create_gmail_get_message_tool(call_tracker)
    gmail_send_tool = create_gmail_send_email_tool(call_tracker)

    # Create mock model for agent node
    model_def = ModelDefinition(
        model_id="openai/gpt-oss-120b",
        chat_model_factory=lambda: MagicMock(),
    )

    # Compile the workflow
    compiled = await compile_workflow(
        EMAIL_TRIAGE_WORKFLOW_SPEC,
        tool_defs=[google_docs_tool, gmail_get_tool, gmail_send_tool],
        model_defs=[model_def],
    )

    # Create trigger envelope matching the spec's trigger definition
    trigger_envelope = {
        "trigger_id": "incoming_email",
        "trigger_key": "poll.gmail.email_received",
        "occurred_at": "2024-01-01T00:00:00Z",
        "message_id": "msg_test_123",
        "thread_id": "thread_test_456",
        "internal_date_ms": 1704067200000,
        "subject": "Question about your product",
        "snippet": "Hello, I have a question...",
        "from": {"name": "Test User", "email": "testuser@example.com"},
        "to": [{"name": "Support", "email": "support@company.com"}],
        "labels": ["INBOX", "UNREAD"],
    }

    # Execute the workflow
    with patch("seer.core.nodes.agent_node.create_agent", return_value=_create_inquiry_mock_agent()):
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # ==========================================================================
    # Verification: Tool Execution Order
    # ==========================================================================

    # Both tools should have executed
    google_docs_calls = [c for c in call_tracker if c.startswith("google_docs_read:")]
    gmail_get_calls = [c for c in call_tracker if c.startswith("gmail_get_message:")]
    gmail_send_calls = [c for c in call_tracker if c.startswith("gmail_send_email:")]

    assert len(google_docs_calls) == 1, f"Expected 1 google_docs_read call, got {len(google_docs_calls)}"
    assert len(gmail_get_calls) == 1, f"Expected 1 gmail_get_message call, got {len(gmail_get_calls)}"
    assert len(gmail_send_calls) == 1, f"Expected 1 gmail_send_email call (should_reply=True), got {len(gmail_send_calls)}"

    # ==========================================================================
    # Verification: State Contains All Node Outputs
    # ==========================================================================

    assert "fetch_product_spec" in result, (
        "fetch_product_spec output missing from state! "
        "This could indicate the tool didn't execute or output wasn't stored."
    )
    assert "fetch_full_email" in result, (
        "fetch_full_email output missing from state! "
        "This could indicate the tool didn't execute or output wasn't stored."
    )
    assert "analyze_and_generate_response" in result, (
        "analyze_and_generate_response output missing from state! "
        "This indicates the LLM node didn't execute properly."
    )
    assert "send_reply" in result, (
        "send_reply output missing from state! "
        "The conditional branch should have taken the true path since should_reply=True."
    )

    # ==========================================================================
    # Verification: Tool Output Content
    # ==========================================================================

    # fetch_product_spec should have returned our mock data
    assert result["fetch_product_spec"]["title"] == "Product Specification"
    assert "AI-powered email assistant" in result["fetch_product_spec"]["body"]

    # fetch_full_email should have returned our mock email
    assert result["fetch_full_email"]["subject"] == "Question about your product"
    assert "automatic triage" in result["fetch_full_email"]["body"]

    # LLM output should match our mock response
    assert result["analyze_and_generate_response"]["classification"] == "inquiry"
    assert result["analyze_and_generate_response"]["should_reply"] is True

    # send_reply should have been called with correct inputs
    assert result["send_reply"]["status"] == "sent"


@pytest.mark.asyncio
async def test_email_triage_workflow_no_reply_branch() -> None:
    """
    Test the workflow's conditional false branch when should_reply=False.

    Verifies that send_reply does NOT execute when the LLM determines
    no reply is needed.
    """
    call_tracker: List[str] = []

    # Create mock tools
    google_docs_tool = create_google_docs_read_tool(call_tracker)
    gmail_get_tool = create_gmail_get_message_tool(call_tracker)
    gmail_send_tool = create_gmail_send_email_tool(call_tracker)

    # Create mock model for agent node that returns should_reply=False
    model_def = ModelDefinition(
        model_id="openai/gpt-oss-120b",
        chat_model_factory=lambda: MagicMock(),
    )

    compiled = await compile_workflow(
        EMAIL_TRIAGE_WORKFLOW_SPEC,
        tool_defs=[google_docs_tool, gmail_get_tool, gmail_send_tool],
        model_defs=[model_def],
    )

    trigger_envelope = {
        "trigger_id": "incoming_email",
        "trigger_key": "poll.gmail.email_received",
        "message_id": "spam_msg_123",
        "thread_id": "spam_thread_456",
        "internal_date_ms": 1704067200000,
        "subject": "You've won a prize!",
        "snippet": "Click here to claim...",
        "from": {"name": "Spammer", "email": "spammer@example.com"},
        "to": [{"name": "Support", "email": "support@company.com"}],
        "labels": ["INBOX", "SPAM"],
    }

    with patch("seer.core.nodes.agent_node.create_agent", return_value=_create_no_reply_mock_agent()):
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify both fetch tools executed
    google_docs_calls = [c for c in call_tracker if c.startswith("google_docs_read:")]
    gmail_get_calls = [c for c in call_tracker if c.startswith("gmail_get_message:")]
    gmail_send_calls = [c for c in call_tracker if c.startswith("gmail_send_email:")]

    assert len(google_docs_calls) == 1, "google_docs_read should have executed"
    assert len(gmail_get_calls) == 1, "gmail_get_message should have executed"
    assert len(gmail_send_calls) == 0, "gmail_send_email should NOT execute when should_reply=False"

    # Verify LLM classified as spam
    assert result["analyze_and_generate_response"]["classification"] == "spam"
    assert result["analyze_and_generate_response"]["should_reply"] is False

    # send_reply should NOT be in result since it didn't execute
    assert "send_reply" not in result, "send_reply should not be in state when should_reply=False"


@pytest.mark.asyncio
async def test_email_triage_parallel_tool_execution() -> None:
    """
    Test that the two fetch tools can execute in parallel.

    The workflow spec has both fetch_product_spec and fetch_full_email
    triggered directly from incoming_email, meaning they should be able
    to run in parallel (no dependencies between them).

    This test verifies the graph topology allows parallel execution.
    """
    call_tracker: List[str] = []

    google_docs_tool = create_google_docs_read_tool(call_tracker)
    gmail_get_tool = create_gmail_get_message_tool(call_tracker)
    gmail_send_tool = create_gmail_send_email_tool(call_tracker)

    model_def = ModelDefinition(
        model_id="openai/gpt-oss-120b",
        chat_model_factory=lambda: MagicMock(),
    )

    compiled = await compile_workflow(
        EMAIL_TRIAGE_WORKFLOW_SPEC,
        tool_defs=[google_docs_tool, gmail_get_tool, gmail_send_tool],
        model_defs=[model_def],
    )

    trigger_envelope = {
        "trigger_id": "incoming_email",
        "trigger_key": "poll.gmail.email_received",
        "message_id": "msg_parallel_test",
        "thread_id": "thread_parallel",
        "internal_date_ms": 1704067200000,
        "subject": "Test",
        "snippet": "Test",
        "from": {"name": "User", "email": "user@example.com"},
        "to": [{"name": "Support", "email": "support@company.com"}],
        "labels": ["INBOX"],
    }

    with patch("seer.core.nodes.agent_node.create_agent", return_value=_create_inquiry_mock_agent()):
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Both tools should have executed
    assert "fetch_product_spec" in result
    assert "fetch_full_email" in result

    # The analyze_and_generate_response node depends on both
    # If it executed, both fetch tools must have completed first
    assert "analyze_and_generate_response" in result

    # This verifies the graph correctly waits for both parallel branches
    # before executing the LLM node


@pytest.mark.asyncio
async def test_email_triage_trigger_data_access() -> None:
    """
    Test that trigger data is correctly accessible via ${incoming_email.*}.

    This specifically tests that expressions like ${incoming_email.message_id}
    resolve correctly during execution.
    """
    call_tracker: List[str] = []

    google_docs_tool = create_google_docs_read_tool(call_tracker)
    gmail_get_tool = create_gmail_get_message_tool(call_tracker)
    gmail_send_tool = create_gmail_send_email_tool(call_tracker)

    model_def = ModelDefinition(
        model_id="openai/gpt-oss-120b",
        chat_model_factory=lambda: MagicMock(),
    )

    compiled = await compile_workflow(
        EMAIL_TRIAGE_WORKFLOW_SPEC,
        tool_defs=[google_docs_tool, gmail_get_tool, gmail_send_tool],
        model_defs=[model_def],
    )

    # Use a specific message_id to verify it's passed correctly
    trigger_envelope = {
        "trigger_id": "incoming_email",
        "trigger_key": "poll.gmail.email_received",
        "message_id": "unique_msg_id_12345",
        "thread_id": "unique_thread_67890",
        "internal_date_ms": 1704067200000,
        "subject": "Trigger Data Test",
        "snippet": "Testing trigger data access",
        "from": {"name": "Trigger Test", "email": "trigger@test.com"},
        "to": [{"name": "Support", "email": "support@company.com"}],
        "labels": ["INBOX"],
    }

    with patch("seer.core.nodes.agent_node.create_agent", return_value=_create_inquiry_mock_agent()):
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify gmail_get_message received the correct message_id from trigger
    gmail_get_calls = [c for c in call_tracker if c.startswith("gmail_get_message:")]
    assert len(gmail_get_calls) == 1
    assert "unique_msg_id_12345" in gmail_get_calls[0], (
        f"Expected gmail_get_message to receive message_id from trigger. Got: {gmail_get_calls[0]}"
    )


@pytest.mark.asyncio
async def test_email_triage_state_accumulation() -> None:
    """
    Test that state accumulates correctly through all workflow stages.

    Verifies that:
    1. Tool outputs are stored in state by node ID
    2. LLM output is stored in state
    3. All outputs are available in the final result
    """
    call_tracker: List[str] = []

    google_docs_tool = create_google_docs_read_tool(call_tracker)
    gmail_get_tool = create_gmail_get_message_tool(call_tracker)
    gmail_send_tool = create_gmail_send_email_tool(call_tracker)

    model_def = ModelDefinition(
        model_id="openai/gpt-oss-120b",
        chat_model_factory=lambda: MagicMock(),
    )

    compiled = await compile_workflow(
        EMAIL_TRIAGE_WORKFLOW_SPEC,
        tool_defs=[google_docs_tool, gmail_get_tool, gmail_send_tool],
        model_defs=[model_def],
    )

    trigger_envelope = {
        "trigger_id": "incoming_email",
        "trigger_key": "poll.gmail.email_received",
        "message_id": "state_test_msg",
        "thread_id": "state_test_thread",
        "internal_date_ms": 1704067200000,
        "subject": "State Accumulation Test",
        "snippet": "Testing state",
        "from": {"name": "User", "email": "user@test.com"},
        "to": [{"name": "Support", "email": "support@company.com"}],
        "labels": ["INBOX"],
    }

    with patch("seer.core.nodes.agent_node.create_agent", return_value=_create_inquiry_mock_agent()):
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify all expected node outputs are in final state
    expected_nodes = [
        "fetch_product_spec",
        "fetch_full_email",
        "analyze_and_generate_response",
        "check_should_reply",  # If node should have a trace
        "send_reply",  # Because should_reply=True
    ]

    for node_id in expected_nodes:
        if node_id == "check_should_reply":
            # If nodes may not produce output in state
            continue
        assert node_id in result, f"Expected node '{node_id}' output in final state, but it's missing"

    # Verify trace keys exist
    trace_keys = [k for k in result.keys() if k.startswith("_trace_")]
    assert len(trace_keys) > 0, "Expected trace keys in result for debugging"


@pytest.mark.asyncio
async def test_email_triage_llm_receives_tool_outputs() -> None:
    """
    Test that the LLM node receives tool outputs in its prompt.

    This test uses a custom LLM handler that captures the invocation
    to verify the prompt contains resolved tool outputs.
    """
    call_tracker: List[str] = []
    agent_invocations: List[Dict[str, Any]] = []

    google_docs_tool = create_google_docs_read_tool(call_tracker)
    gmail_get_tool = create_gmail_get_message_tool(call_tracker)
    gmail_send_tool = create_gmail_send_email_tool(call_tracker)

    model_def = ModelDefinition(
        model_id="openai/gpt-oss-120b",
        chat_model_factory=lambda: MagicMock(),
    )

    compiled = await compile_workflow(
        EMAIL_TRIAGE_WORKFLOW_SPEC,
        tool_defs=[google_docs_tool, gmail_get_tool, gmail_send_tool],
        model_defs=[model_def],
    )

    trigger_envelope = {
        "trigger_id": "incoming_email",
        "trigger_key": "poll.gmail.email_received",
        "message_id": "llm_input_test",
        "thread_id": "llm_thread",
        "internal_date_ms": 1704067200000,
        "subject": "LLM Input Test",
        "snippet": "Testing LLM input",
        "from": {"name": "LLM Tester", "email": "llm@test.com"},
        "to": [{"name": "Support", "email": "support@company.com"}],
        "labels": ["INBOX"],
    }

    # Create a capturing mock agent that records its inputs
    mock_agent = AsyncMock()

    async def capturing_ainvoke(inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Capture the agent invocation for inspection."""
        agent_invocations.append(inputs)
        return {"messages": [AIMessage(content=json.dumps(_INQUIRY_RESPONSE))]}

    mock_agent.ainvoke = capturing_ainvoke

    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify agent was invoked
    assert len(agent_invocations) == 1, "Expected exactly one agent invocation"

    # The prompt should contain resolved tool outputs (first message is the HumanMessage with rendered prompt)
    first_message = agent_invocations[0].get("messages", [{}])[0]
    prompt = first_message.content if hasattr(first_message, "content") else ""

    # These strings come from our mock tool outputs
    assert "Product Specification" in prompt or "AI-powered email assistant" in prompt, (
        f"Agent prompt should contain fetch_product_spec output. Got prompt: {prompt[:500]}..."
    )
    assert "Question about your product" in prompt or "automatic triage" in prompt or "LLM Input Test" in prompt, (
        f"Agent prompt should contain fetch_full_email output. Got prompt: {prompt[:500]}..."
    )


# =============================================================================
# MULTI-TRIGGER CONVERGING WORKFLOW TESTS
# =============================================================================
# Test scenario: Different triggers lead to the same merge node through separate paths.
# In this case, the merge node should NOT wait for all branches - only the active one.
#
# Workflow structure:
#   trigger_a -> node_c -> node_m
#   trigger_b -> node_d -> node_m
#
# When trigger_a fires, only node_c runs, then node_m runs (without waiting for node_d).
# When trigger_b fires, only node_d runs, then node_m runs (without waiting for node_c).


MULTI_TRIGGER_CONVERGING_SPEC: Dict[str, Any] = {
    "version": "2",
    "triggers": [
        {
            "id": "trigger_a",
            "key": "webhook.event_a",
            "mode": "webhook",
            "event_schema": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                },
            },
        },
        {
            "id": "trigger_b",
            "key": "webhook.event_b",
            "mode": "webhook",
            "event_schema": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                },
            },
        },
    ],
    "nodes": [
        {
            "id": "node_c",
            "type": "tool",
            "tool": "test.tracker",
            "inputs": {"value": "from_trigger_a_${trigger_a.value}"},
        },
        {
            "id": "node_d",
            "type": "tool",
            "tool": "test.tracker",
            "inputs": {"value": "from_trigger_b_${trigger_b.value}"},
        },
        {
            "id": "node_m",
            "type": "tool",
            "tool": "test.tracker",
            "inputs": {"value": "merged"},
        },
    ],
    "edges": [
        {"source": "trigger_a", "target": "node_c", "type": "trigger"},
        {"source": "trigger_b", "target": "node_d", "type": "trigger"},
        {"source": "node_c", "target": "node_m", "type": "default"},
        {"source": "node_d", "target": "node_m", "type": "default"},
    ],
}


def create_simple_tracking_tool(call_tracker: List[str]) -> ToolDefinition:
    """Create a simple tracking tool for multi-trigger tests."""

    def handler(inputs: Dict[str, Any], _config: Any, _context: Any) -> str:
        value = inputs.get("value", "unknown")
        call_tracker.append(value)
        return value

    async def async_handler(inputs: Dict[str, Any], _config: Any, _context: Any) -> str:
        value = inputs.get("value", "unknown")
        call_tracker.append(value)
        return value

    return ToolDefinition(
        name="test.tracker",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        output_schema={"type": "string"},
        handler=handler,
        async_handler=async_handler,
    )


@pytest.mark.asyncio
async def test_multi_trigger_converging_trigger_a_path() -> None:
    """
    Test that when trigger_a fires, only node_c -> node_m executes.

    The merge node (node_m) should NOT wait for node_d since trigger_b
    didn't fire. This tests that our convergence detection correctly
    handles branches from DIFFERENT triggers.

    Workflow:
        trigger_a -> node_c -> node_m  (THIS PATH SHOULD EXECUTE)
        trigger_b -> node_d -> node_m  (THIS PATH SHOULD NOT EXECUTE)
    """
    call_tracker: List[str] = []
    tracking_tool = create_simple_tracking_tool(call_tracker)

    compiled = await compile_workflow(
        MULTI_TRIGGER_CONVERGING_SPEC,
        tool_defs=[tracking_tool],
    )

    # Fire trigger_a
    trigger_envelope = {
        "trigger_id": "trigger_a",
        "trigger_key": "webhook.event_a",
        "value": "hello_a",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify node_c executed (from trigger_a path)
    assert "from_trigger_a_hello_a" in call_tracker, (
        f"node_c should have executed with trigger_a data. Got: {call_tracker}"
    )

    # Verify node_m executed (the merge node)
    assert "merged" in call_tracker, (
        f"node_m should have executed after node_c. Got: {call_tracker}"
    )

    # Verify node_d did NOT execute (trigger_b path)
    node_d_calls = [c for c in call_tracker if c.startswith("from_trigger_b")]
    assert len(node_d_calls) == 0, (
        f"node_d should NOT have executed since trigger_b didn't fire. Got: {call_tracker}"
    )

    # Verify execution order
    node_c_idx = call_tracker.index("from_trigger_a_hello_a")
    node_m_idx = call_tracker.index("merged")
    assert node_c_idx < node_m_idx, "node_c should execute before node_m"

    # Verify state contains correct outputs
    assert "node_c" in result, "node_c output should be in state"
    assert "node_m" in result, "node_m output should be in state"
    assert result["node_c"] == "from_trigger_a_hello_a"
    assert result["node_m"] == "merged"


@pytest.mark.asyncio
async def test_multi_trigger_converging_trigger_b_path() -> None:
    """
    Test that when trigger_b fires, only node_d -> node_m executes.

    The merge node (node_m) should NOT wait for node_c since trigger_a
    didn't fire.

    Workflow:
        trigger_a -> node_c -> node_m  (THIS PATH SHOULD NOT EXECUTE)
        trigger_b -> node_d -> node_m  (THIS PATH SHOULD EXECUTE)
    """
    call_tracker: List[str] = []
    tracking_tool = create_simple_tracking_tool(call_tracker)

    compiled = await compile_workflow(
        MULTI_TRIGGER_CONVERGING_SPEC,
        tool_defs=[tracking_tool],
    )

    # Fire trigger_b
    trigger_envelope = {
        "trigger_id": "trigger_b",
        "trigger_key": "webhook.event_b",
        "value": "hello_b",
    }

    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify node_d executed (from trigger_b path)
    assert "from_trigger_b_hello_b" in call_tracker, (
        f"node_d should have executed with trigger_b data. Got: {call_tracker}"
    )

    # Verify node_m executed (the merge node)
    assert "merged" in call_tracker, (
        f"node_m should have executed after node_d. Got: {call_tracker}"
    )

    # Verify node_c did NOT execute (trigger_a path)
    node_c_calls = [c for c in call_tracker if c.startswith("from_trigger_a")]
    assert len(node_c_calls) == 0, (
        f"node_c should NOT have executed since trigger_a didn't fire. Got: {call_tracker}"
    )

    # Verify execution order
    node_d_idx = call_tracker.index("from_trigger_b_hello_b")
    node_m_idx = call_tracker.index("merged")
    assert node_d_idx < node_m_idx, "node_d should execute before node_m"

    # Verify state contains correct outputs
    assert "node_d" in result, "node_d output should be in state"
    assert "node_m" in result, "node_m output should be in state"
    assert result["node_d"] == "from_trigger_b_hello_b"
    assert result["node_m"] == "merged"


@pytest.mark.asyncio
async def test_multi_trigger_converging_both_triggers_independently() -> None:
    """
    Test that both triggers work independently and produce correct results.

    This verifies that the workflow compiles correctly and each trigger
    path functions in isolation.
    """
    call_tracker: List[str] = []
    tracking_tool = create_simple_tracking_tool(call_tracker)

    compiled = await compile_workflow(
        MULTI_TRIGGER_CONVERGING_SPEC,
        tool_defs=[tracking_tool],
    )

    # First: Fire trigger_a
    result_a = await compiled.ainvoke(
        config=None,
        context=None,
        trigger={
            "trigger_id": "trigger_a",
            "trigger_key": "webhook.event_a",
            "value": "test_a",
        },
    )

    # Clear tracker for second run
    call_tracker.clear()

    # Second: Fire trigger_b
    result_b = await compiled.ainvoke(
        config=None,
        context=None,
        trigger={
            "trigger_id": "trigger_b",
            "trigger_key": "webhook.event_b",
            "value": "test_b",
        },
    )

    # Verify trigger_a path produced correct results
    assert "node_c" in result_a
    assert "node_m" in result_a
    assert result_a["node_c"] == "from_trigger_a_test_a"

    # Verify trigger_b path produced correct results
    assert "node_d" in result_b
    assert "node_m" in result_b
    assert result_b["node_d"] == "from_trigger_b_test_b"
