"""
Message sanitization utilities for API compatibility.

Handles malformed tool_call_id values from LLM providers (e.g., OpenRouter/Kimi)
that may have leading/trailing whitespace.
"""
import uuid
from typing import Any, List, Optional


def sanitize_tool_call_id(tool_call_id: Optional[str]) -> str:
    """
    Sanitize tool_call_id by stripping leading/trailing whitespace.

    Handles malformed IDs from OpenRouter/Kimi models that include
    leading/trailing whitespace (e.g., " list_available_tools:8").

    Args:
        tool_call_id: The raw tool_call_id from LLM response

    Returns:
        Sanitized tool_call_id with whitespace stripped
    """
    if not tool_call_id:
        return f"tool_call_{uuid.uuid4().hex[:8]}"

    # Only strip leading/trailing whitespace
    sanitized = tool_call_id.strip()

    return sanitized or f"tool_call_{uuid.uuid4().hex[:8]}"


def _sanitize_dict_tool_call(tc: dict) -> None:
    """Sanitize a dict-based tool call's id field."""
    if 'id' in tc:
        tc['id'] = sanitize_tool_call_id(tc['id'])


def _sanitize_object_tool_call(tc: Any) -> None:
    """Sanitize an object-based tool call's id attribute."""
    tool_id = getattr(tc, 'id', None)
    if not tool_id:
        return
    try:
        setattr(tc, 'id', sanitize_tool_call_id(tool_id))
    except AttributeError:
        # Object may be immutable; skip
        pass


def _sanitize_tool_call(tc: Any) -> None:
    """Sanitize a single tool call (dict or object)."""
    if isinstance(tc, dict):
        _sanitize_dict_tool_call(tc)
    elif hasattr(tc, 'id'):
        _sanitize_object_tool_call(tc)


def _sanitize_message_tool_call_id(msg: Any) -> None:
    """Sanitize a message's tool_call_id attribute."""
    tool_call_id = getattr(msg, 'tool_call_id', None)
    if not tool_call_id:
        return
    try:
        setattr(msg, 'tool_call_id', sanitize_tool_call_id(tool_call_id))
    except AttributeError:
        # Message may be immutable; skip
        pass


def sanitize_messages_tool_call_ids(messages: List[Any]) -> List[Any]:
    """
    Sanitize tool_call_ids in message history before sending to LLM.

    Modifies messages in-place to clean tool_call_ids in:
    - AIMessage.tool_calls[].id
    - ToolMessage.tool_call_id

    Args:
        messages: List of LangChain message objects

    Returns:
        The same list with sanitized tool_call_ids
    """
    for msg in messages:
        # Sanitize AIMessage tool_calls
        tool_calls = getattr(msg, 'tool_calls', None)
        if tool_calls:
            for tc in tool_calls:
                _sanitize_tool_call(tc)

        # Sanitize ToolMessage tool_call_id
        if hasattr(msg, 'tool_call_id'):
            _sanitize_message_tool_call_id(msg)

    return messages
