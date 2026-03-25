"""Standalone image generation utility via OpenRouter API."""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import httpx

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)


async def generate_image(prompt: str, model: str = "sourceful/riverflow-v2-fast", size: str = "1024x1024") -> Tuple[str, Dict[str, Any]]:
    """
    Generate an image via OpenRouter API.

    Returns tuple of (image_url, usage_metadata).
    Raises ValueError if API key not configured or generation fails.
    """
    api_key = config.openrouter_api_key
    if not api_key:
        raise ValueError("OpenRouter API key not configured")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://seer.app",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if size:
        width, _, height = size.partition("x")
        if width.isdigit() and height.isdigit():
            payload["provider"] = {"sort": "throughput"}
            payload["metadata"] = {"image_size": size, "width": int(width), "height": int(height)}

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise ValueError(f"OpenRouter API error (status {response.status_code}): {response.text}")

    data = response.json()
    image_url = _extract_image_url(data)

    usage = data.get("usage", {})
    usage_metadata = {
        "model": data.get("model", model),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }

    return image_url, usage_metadata


def _extract_from_images(images: list) -> str:
    """Extract URL from message.images array."""
    for img in images:
        if not isinstance(img, dict):
            continue
        if img.get("type") == "image_url":
            url = img.get("image_url", {}).get("url", "")
            if url:
                return url
        elif img.get("url"):
            return img["url"]
    return ""


def _extract_from_content(content) -> str:
    """Extract URL from message.content (string or list)."""
    if isinstance(content, str):
        if content.startswith("data:"):
            return content
        url_match = re.search(r'https?://[^\s\)\"\']+', content)
        return url_match.group(0) if url_match else ""
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image_url":
                return block.get("image_url", {}).get("url", "")
            if block.get("type") == "image":
                return block.get("url", "") or block.get("image_url", "")
    return ""


def _extract_image_url(data: Dict[str, Any]) -> str:
    """Extract image URL from OpenRouter response."""
    choices = data.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})

    images = message.get("images", [])
    if images and isinstance(images, list):
        url = _extract_from_images(images)
        if url:
            return url

    return _extract_from_content(message.get("content", ""))
