from __future__ import annotations
from typing import Any, Mapping
from langchain_core.messages import AIMessage



def message_to_text(message: Any | AIMessage) -> str:
    """
    LangChain responses can return strings, AIMessage objects, or richer payloads.
    This helper normalizes them into a plain string for downstream processing.
    """

    if message is None:
        return ""

    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            # Skip reasoning blocks and other non-text content types
        return "".join(parts)
    return str(content)
