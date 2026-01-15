"""
Credit calculator for converting LLM token usage to USD costs.

Pricing updated: January 2026
"""
from decimal import Decimal
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing per million tokens for a model."""

    input_per_1m: Decimal
    output_per_1m: Decimal
    reasoning_per_1m: Optional[Decimal] = None  # For o3 and reasoning models


# Pricing registry (per 1M tokens in USD)
MODEL_PRICING: Dict[str, ModelPricing] = {
    # OpenAI GPT-5 series
    "gpt-5": ModelPricing(
        input_per_1m=Decimal("5.00"),
        output_per_1m=Decimal("15.00"),
    ),
    "gpt-5-mini": ModelPricing(
        input_per_1m=Decimal("0.200"),
        output_per_1m=Decimal("0.800"),
    ),
    "gpt-5-nano": ModelPricing(
        input_per_1m=Decimal("0.050"),
        output_per_1m=Decimal("0.200"),
    ),

    # "gpt-5-nano": ModelPricing(
    #     input_per_1m=Decimal("500"),
    #     output_per_1m=Decimal("2000"),
    # ),
    # OpenAI GPT-4 series
    "gpt-4o": ModelPricing(
        input_per_1m=Decimal("2.50"),
        output_per_1m=Decimal("10.00"),
    ),
    "gpt-4o-mini": ModelPricing(
        input_per_1m=Decimal("0.150"),
        output_per_1m=Decimal("0.600"),
    ),
    "gpt-4-turbo": ModelPricing(
        input_per_1m=Decimal("10.00"),
        output_per_1m=Decimal("30.00"),
    ),
    "gpt-4": ModelPricing(
        input_per_1m=Decimal("30.00"),
        output_per_1m=Decimal("60.00"),
    ),
    # OpenAI o3 reasoning models (reasoning tokens billed as output)
    "o3-mini": ModelPricing(
        input_per_1m=Decimal("1.10"),
        output_per_1m=Decimal("4.40"),
    ),
    "o3": ModelPricing(
        input_per_1m=Decimal("2.00"),
        output_per_1m=Decimal("8.00"),
    ),
    # Anthropic Claude 4.5 series
    "claude-sonnet-4.5": ModelPricing(
        input_per_1m=Decimal("3.00"),
        output_per_1m=Decimal("15.00"),
    ),
    "claude-opus-4.5": ModelPricing(
        input_per_1m=Decimal("5.00"),
        output_per_1m=Decimal("25.00"),
    ),
    "claude-haiku-4.5": ModelPricing(
        input_per_1m=Decimal("1.00"),
        output_per_1m=Decimal("5.00"),
    ),
    # Anthropic Claude 3 series
    "claude-3-opus-20240229": ModelPricing(
        input_per_1m=Decimal("15.00"),
        output_per_1m=Decimal("75.00"),
    ),
    "claude-3-sonnet-20240229": ModelPricing(
        input_per_1m=Decimal("3.00"),
        output_per_1m=Decimal("15.00"),
    ),
    "claude-3-5-sonnet-20241022": ModelPricing(
        input_per_1m=Decimal("3.00"),
        output_per_1m=Decimal("15.00"),
    ),
    "claude-3-5-sonnet-20240620": ModelPricing(
        input_per_1m=Decimal("3.00"),
        output_per_1m=Decimal("15.00"),
    ),
    "claude-3-haiku-20240307": ModelPricing(
        input_per_1m=Decimal("0.25"),
        output_per_1m=Decimal("1.25"),
    ),
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
) -> Decimal:
    """
    Calculate USD cost for an LLM API call.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4.5")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        reasoning_tokens: Number of reasoning tokens (o3 models, Claude extended thinking)

    Returns:
        Cost in USD as Decimal

    Raises:
        ValueError: If model not found in pricing registry
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        # Fallback: use gpt-4o pricing for unknown models
        pricing = MODEL_PRICING["gpt-4o"]

    # Calculate costs per token type
    input_cost = (Decimal(input_tokens) / Decimal(1_000_000)) * pricing.input_per_1m
    output_cost = (Decimal(output_tokens) / Decimal(1_000_000)) * pricing.output_per_1m

    # Reasoning tokens are billed as output tokens
    if reasoning_tokens > 0:
        reasoning_cost = (Decimal(reasoning_tokens) / Decimal(1_000_000)) * pricing.output_per_1m
    else:
        reasoning_cost = Decimal("0")

    total_cost = input_cost + output_cost + reasoning_cost

    # Round to 6 decimal places (matches DB schema)
    return total_cost.quantize(Decimal("0.000001"))
