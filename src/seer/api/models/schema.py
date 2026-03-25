"""Pydantic schemas for model information."""
from typing import Literal

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Information about an available LLM model."""
    id: str  # Model identifier (e.g., "qwen/qwen3-235b-a22b-2507", "moonshotai/kimi-k2.5")
    provider: Literal["openai", "anthropic", "openrouter"]
    name: str  # Display name (e.g., "Qwen 3", "Kimi")
    available: bool  # Whether API key is configured
