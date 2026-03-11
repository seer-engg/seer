"""Pydantic schemas for model information."""
from typing import Literal

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Information about an available LLM model."""
    id: str  # Model identifier (e.g., "openai/gpt-oss-120b", "claude-opus-4-5")
    provider: Literal["openai", "anthropic", "openrouter"]
    name: str  # Display name (e.g., "GPT OSS 120B", "Claude Opus 4.5")
    available: bool  # Whether API key is configured
