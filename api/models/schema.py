"""Pydantic schemas for model information."""
from typing import Literal
from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Information about an available LLM model."""
    id: str  # Model identifier (e.g., "gpt-5.2", "claude-opus-4-5")
    provider: Literal["openai", "anthropic"]
    name: str  # Display name (e.g., "GPT-5.2", "Claude Opus 4.5")
    available: bool  # Whether API key is configured

