"""Shared utilities for the evaluation agent."""
from typing import Any


def normalize_raw_request(raw_request: Any) -> str:
    """
    Normalize the raw_request field into a plain string.
    Handles cases where the request is a list of message chunks or dicts.
    """
    if not raw_request:
        return ""

    if isinstance(raw_request, str):
        return raw_request

    if isinstance(raw_request, list):
        parts: list[str] = []
        for chunk in raw_request:
            if isinstance(chunk, dict):
                if "text" in chunk and isinstance(chunk["text"], str):
                    parts.append(chunk["text"])
                elif "content" in chunk and isinstance(chunk["content"], str):
                    parts.append(chunk["content"])
                else:
                    parts.append(str(chunk))
            else:
                parts.append(str(chunk))
        return "\n".join(filter(None, parts))

    return str(raw_request)

