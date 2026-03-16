from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultModelDescriptor:
    """Shared default model metadata used by compiler and API catalog layers."""

    id: str
    title: str
    supports_json_schema: bool = True


DEFAULT_AGENT_MODELS = (
    DefaultModelDescriptor(id="openai/gpt-oss-120b", title="GPT OSS 120B", supports_json_schema=True),
    DefaultModelDescriptor(id="z-ai/glm-5", title="GLM 5", supports_json_schema=True),
    DefaultModelDescriptor(id="moonshotai/kimi-k2.5", title="Kimi K2.5", supports_json_schema=True),
    DefaultModelDescriptor(id="minimax/minimax-m2.5", title="MiniMax M2.5", supports_json_schema=True),
)

DEFAULT_AGENT_MODEL_IDS = frozenset(model.id for model in DEFAULT_AGENT_MODELS)
