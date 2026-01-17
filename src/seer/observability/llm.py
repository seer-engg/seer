from __future__ import annotations
from typing import Any, Dict




def extract_usage_metadata(response: Any, model_id: str) -> Dict[str, Any]:
    """
    Extract token usage metadata from LangChain response.

    Args:
        response: LangChain AIMessage response
        model_id: Model identifier

    Returns:
        Dictionary with keys: input_tokens, output_tokens, reasoning_tokens, model
    """
    usage_meta = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "model": model_id,
    }

    # Try usage_metadata first (newer LangChain)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage_meta["input_tokens"] = response.usage_metadata.get("input_tokens", 0)
        usage_meta["output_tokens"] = response.usage_metadata.get("output_tokens", 0)
        # Reasoning tokens might be in different fields
        usage_meta["reasoning_tokens"] = response.usage_metadata.get("reasoning_tokens", 0)
        return usage_meta

    # Fallback to response_metadata (older format)
    if hasattr(response, "response_metadata") and response.response_metadata:
        token_usage = response.response_metadata.get("token_usage", {})
        usage_meta["input_tokens"] = token_usage.get("prompt_tokens", 0)
        usage_meta["output_tokens"] = token_usage.get("completion_tokens", 0)
        # Some models include reasoning_tokens or cached_tokens
        usage_meta["reasoning_tokens"] = token_usage.get("reasoning_tokens", 0)

    return usage_meta
