"""
Unit tests for ChatOrchestrator streaming methods.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from seer.api.agents.workflow.chat_schema import StreamEventType
from seer.api.agents.workflow.chat_services import ChatOrchestrator


def _make_orchestrator(agent=None, checkpointer=None):
    """Helper to build a minimal ChatOrchestrator for testing."""
    return ChatOrchestrator(
        agent=agent or AsyncMock(),
        checkpointer=checkpointer,
        health_service=MagicMock(),
        detector=MagicMock(),
        recovery_service=MagicMock(),
        reconnect_func=AsyncMock(),
    )


def _make_publisher():
    """Mock StreamPublisher."""
    pub = AsyncMock()
    pub.published_events = []

    async def record_publish(event_type, data):
        pub.published_events.append((event_type, data))
        return "1234-0"

    pub.publish = record_publish
    return pub


@pytest.mark.asyncio
async def test_stream_with_timeout_publishes_tool_events():
    """stream_with_timeout should emit TOOL_START and TOOL_END for tool calls."""
    events = [
        {"event": "on_tool_start", "name": "search_web", "data": {"input": {"query": "test"}}, "run_id": "r1"},
        {"event": "on_tool_end", "name": "search_web", "data": {"output": "result data"}, "run_id": "r1"},
        {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {"messages": []}}, "run_id": "r0"},
    ]

    async def fake_astream_events(*args, **kwargs):
        for e in events:
            yield e

    mock_agent = MagicMock()
    mock_agent.astream_events = fake_astream_events

    orchestrator = _make_orchestrator(agent=mock_agent)
    publisher = _make_publisher()

    with patch("seer.agents.nexus._current_thread_id") as mock_tid:
        mock_tid.set = MagicMock(return_value=None)
        mock_tid.reset = MagicMock()

        result = await orchestrator.stream_with_timeout(
            {"messages": []},
            {"configurable": {"thread_id": "t1"}, "recursion_limit": 10},
            publisher,
        )

    # Should have published TOOL_START and TOOL_END
    event_types = [e[0] for e in publisher.published_events]
    assert StreamEventType.TOOL_START in event_types
    assert StreamEventType.TOOL_END in event_types

    # TOOL_START should have tool_name
    tool_start = next(e for e in publisher.published_events if e[0] == StreamEventType.TOOL_START)
    assert tool_start[1]["tool_name"] == "search_web"
    assert "input_preview" in tool_start[1]

    tool_end = next(e for e in publisher.published_events if e[0] == StreamEventType.TOOL_END)
    assert tool_end[1]["tool_name"] == "search_web"


@pytest.mark.asyncio
async def test_stream_with_timeout_captures_final_content():
    """stream_with_timeout should capture final AI message as final_content."""
    # Simulate: one tool-calling step, then a terminal message
    ai_with_tool_calls = MagicMock(spec=AIMessage)
    ai_with_tool_calls.tool_calls = [{"id": "tc1", "name": "foo", "args": {}}]
    ai_with_tool_calls.content = "Calling tool..."

    ai_final = MagicMock(spec=AIMessage)
    ai_final.tool_calls = []
    ai_final.content = "Here is the final answer!"

    events = [
        {"event": "on_chat_model_end", "name": "ChatModel", "data": {"output": ai_with_tool_calls}, "run_id": "r1"},
        {"event": "on_chat_model_end", "name": "ChatModel", "data": {"output": ai_final}, "run_id": "r2"},
        {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {"messages": [ai_final]}}, "run_id": "r0"},
    ]

    async def fake_astream_events(*args, **kwargs):
        for e in events:
            yield e

    mock_agent = MagicMock()
    mock_agent.astream_events = fake_astream_events

    orchestrator = _make_orchestrator(agent=mock_agent)
    publisher = _make_publisher()

    with patch("seer.agents.nexus._current_thread_id") as mock_tid:
        mock_tid.set = MagicMock(return_value=None)
        mock_tid.reset = MagicMock()

        result = await orchestrator.stream_with_timeout(
            {"messages": []},
            {"configurable": {"thread_id": "t1"}, "recursion_limit": 10},
            publisher,
        )

    # final_content should be the terminal message
    assert result["final_content"] == "Here is the final answer!"

    # Intermediate tool-calling message should be published as AI_MESSAGE
    event_types = [e[0] for e in publisher.published_events]
    assert StreamEventType.AI_MESSAGE in event_types


@pytest.mark.asyncio
async def test_stream_with_timeout_raises_on_timeout():
    """stream_with_timeout should raise HTTPException on timeout."""
    import asyncio

    async def slow_stream(*args, **kwargs):
        await asyncio.sleep(9999)
        yield {}

    mock_agent = MagicMock()
    mock_agent.astream_events = slow_stream

    orchestrator = _make_orchestrator(agent=mock_agent)
    publisher = _make_publisher()

    from fastapi import HTTPException

    with patch("seer.agents.nexus._current_thread_id") as mock_tid:
        mock_tid.set = MagicMock(return_value=None)
        mock_tid.reset = MagicMock()

        with pytest.raises(Exception):
            await orchestrator.stream_with_timeout(
                {"messages": []},
                {"configurable": {"thread_id": "t1"}, "recursion_limit": 10},
                publisher,
                timeout=0.01,
            )


@pytest.mark.asyncio
async def test_stream_with_health_checks_uses_streaming_path():
    """stream_with_health_checks should call stream_with_timeout on the happy path."""
    orchestrator = _make_orchestrator()
    orchestrator.checkpointer = None  # Skip health checks

    publisher = _make_publisher()

    streaming_result = {"messages": [], "final_content": "streaming result"}

    from langchain_core.messages import HumanMessage
    with patch.object(orchestrator, "stream_with_timeout", new_callable=AsyncMock, return_value=streaming_result) as mock_stream:
        result = await orchestrator.stream_with_health_checks(
            HumanMessage(content="test"),
            {"configurable": {"thread_id": "t1"}, "recursion_limit": 10},
            publisher,
        )

    mock_stream.assert_called_once()
    assert result == streaming_result


@pytest.mark.asyncio
async def test_stream_with_timeout_uses_config_default_timeout():
    """stream_with_timeout should use config default timeout when none is provided."""
    observed_timeout = None

    async def fake_astream_events(*args, **kwargs):
        yield {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {"messages": []}}, "run_id": "r0"}

    async def fake_wait_for(coro, timeout):
        nonlocal observed_timeout
        observed_timeout = timeout
        return await coro

    mock_agent = MagicMock()
    mock_agent.astream_events = fake_astream_events

    orchestrator = _make_orchestrator(agent=mock_agent)
    publisher = _make_publisher()

    with patch("seer.agents.nexus._current_thread_id") as mock_tid, \
         patch("seer.api.agents.workflow.chat_services.config") as mock_config, \
         patch("seer.api.agents.workflow.chat_services.asyncio.wait_for", side_effect=fake_wait_for):
        mock_tid.set = MagicMock(return_value=None)
        mock_tid.reset = MagicMock()
        mock_config.nexus_chat_timeout_seconds = 2700

        await orchestrator.stream_with_timeout(
            {"messages": []},
            {"configurable": {"thread_id": "t1"}, "recursion_limit": 10},
            publisher,
        )

    assert observed_timeout == 2700.0
