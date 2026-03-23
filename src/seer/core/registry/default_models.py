from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultModelDescriptor:
    """Shared default model metadata used by compiler and API catalog layers."""

    id: str
    title: str
    supports_json_schema: bool = True
    category: str | None = None  # "fast", "smart", "balanced"


DEFAULT_AGENT_MODELS = (
    DefaultModelDescriptor(id="qwen/qwen3-235b-a22b-2507", title="Qwen 3", category="smart"),
    DefaultModelDescriptor(id="google/gemini-2.0-flash-001", title="Gemini", category="fast"),
    DefaultModelDescriptor(id="mistralai/mistral-small-3.2-24b-instruct", title="Mistral", category="balanced"),
    DefaultModelDescriptor(id="moonshotai/kimi-k2.5", title="Kimi"),
)

DEFAULT_AGENT_MODEL_IDS = frozenset(model.id for model in DEFAULT_AGENT_MODELS)
