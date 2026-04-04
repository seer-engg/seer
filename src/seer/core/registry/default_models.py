from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultModelDescriptor:
    """Shared default model metadata used by compiler and API catalog layers."""

    id: str
    title: str
    provider: str = "openrouter"  # openrouter, openai, anthropic, google
    supports_json_schema: bool = True
    category: str | None = None  # "fast", "smart", "balanced", "coding", "reasoning"


# ---------------------------------------------------------------------------
# Curated model catalog — popular models across all providers.
#
# Non-BYOK users see only DEFAULT_AGENT_MODELS (platform defaults).
# BYOK users get provider-specific curated models via MODELS_BY_PROVIDER.
# ---------------------------------------------------------------------------

# Platform defaults — shown to non-BYOK users
DEFAULT_AGENT_MODELS = (
    DefaultModelDescriptor(id="qwen/qwen3-235b-a22b-2507", title="Qwen 3", provider="openrouter", category="smart"),
    DefaultModelDescriptor(id="google/gemini-2.0-flash-001", title="Gemini 2.0 Flash", provider="openrouter", category="fast"),
    DefaultModelDescriptor(id="mistralai/mistral-small-3.2-24b-instruct", title="Mistral Small", provider="openrouter", category="balanced"),
    DefaultModelDescriptor(id="moonshotai/kimi-k2.5", title="Kimi K2.5", provider="openrouter"),
)

# OpenAI models (BYOK: direct OpenAI key; non-BYOK: via OpenRouter)
OPENAI_MODELS = (
    DefaultModelDescriptor(id="gpt-4.1", title="GPT-4.1", provider="openai", category="smart"),
    DefaultModelDescriptor(id="gpt-4.1-mini", title="GPT-4.1 Mini", provider="openai", category="fast"),
    DefaultModelDescriptor(id="gpt-4.1-nano", title="GPT-4.1 Nano", provider="openai", category="fast"),
    DefaultModelDescriptor(id="gpt-4o", title="GPT-4o", provider="openai", category="smart"),
    DefaultModelDescriptor(id="gpt-4o-mini", title="GPT-4o Mini", provider="openai", category="fast"),
    DefaultModelDescriptor(id="o4-mini", title="o4-mini", provider="openai", category="reasoning"),
    DefaultModelDescriptor(id="o3", title="o3", provider="openai", category="reasoning"),
    DefaultModelDescriptor(id="o3-mini", title="o3-mini", provider="openai", category="reasoning"),
)

# Anthropic models (BYOK: direct Anthropic key; non-BYOK: via OpenRouter)
ANTHROPIC_MODELS = (
    DefaultModelDescriptor(id="claude-sonnet-4-5-20250514", title="Claude Sonnet 4.5", provider="anthropic", category="smart"),
    DefaultModelDescriptor(id="claude-opus-4-20250514", title="Claude Opus 4", provider="anthropic", category="smart"),
    DefaultModelDescriptor(id="claude-3-5-sonnet-20241022", title="Claude 3.5 Sonnet", provider="anthropic", category="balanced"),
    DefaultModelDescriptor(id="claude-3-5-haiku-20241022", title="Claude 3.5 Haiku", provider="anthropic", category="fast"),
)

# Google models (BYOK: direct Google key; non-BYOK: via OpenRouter)
GOOGLE_MODELS = (
    DefaultModelDescriptor(id="gemini-2.5-pro-preview-05-06", title="Gemini 2.5 Pro", provider="google", category="smart"),
    DefaultModelDescriptor(id="gemini-2.5-flash-preview-05-20", title="Gemini 2.5 Flash", provider="google", category="fast"),
    DefaultModelDescriptor(id="gemini-2.0-flash", title="Gemini 2.0 Flash", provider="google", category="fast"),
)

# Additional open-source / OpenRouter-only models
OPENROUTER_MODELS = (
    DefaultModelDescriptor(id="deepseek/deepseek-r1", title="DeepSeek R1", provider="openrouter", category="reasoning"),
    DefaultModelDescriptor(id="deepseek/deepseek-chat-v3-0324", title="DeepSeek V3", provider="openrouter", category="smart"),
    DefaultModelDescriptor(id="meta-llama/llama-4-maverick", title="Llama 4 Maverick", provider="openrouter", category="smart"),
    DefaultModelDescriptor(id="meta-llama/llama-4-scout", title="Llama 4 Scout", provider="openrouter", category="balanced"),
    DefaultModelDescriptor(id="qwen/qwen3-30b-a3b", title="Qwen 3 30B", provider="openrouter", category="balanced"),
)

# Provider-keyed lookup — used to build model lists for BYOK users
MODELS_BY_PROVIDER: dict[str, tuple[DefaultModelDescriptor, ...]] = {
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
    "google": GOOGLE_MODELS,
    "openrouter": (*DEFAULT_AGENT_MODELS, *OPENROUTER_MODELS),
}

DEFAULT_AGENT_MODEL_IDS = frozenset(model.id for model in DEFAULT_AGENT_MODELS)

# Legacy models that should be transparently remapped at runtime.
DEPRECATED_MODEL_MAP: dict[str, str] = {
    "openai/gpt-oss-120b": "qwen/qwen3-235b-a22b-2507",
    "z-ai/glm-5": "qwen/qwen3-235b-a22b-2507",
    "gpt-4": "qwen/qwen3-235b-a22b-2507",
    "gpt-5-mini": "qwen/qwen3-235b-a22b-2507",
}
