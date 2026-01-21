"""Tests for credit calculation."""
from decimal import Decimal

import pytest

from seer.observability.credit_calculator import calculate_cost


def test_gpt4o_cost():
    """Test GPT-4o cost calculation."""
    cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
    # Input: 1000 * $2.50/1M = $0.0025
    # Output: 500 * $10/1M = $0.005
    # Total: $0.0075
    assert cost == Decimal("0.007500")


def test_gpt4o_mini_cost():
    """Test GPT-4o-mini cost calculation."""
    cost = calculate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
    # Input: 1000 * $0.150/1M = $0.00015
    # Output: 500 * $0.600/1M = $0.0003
    # Total: $0.00045
    assert cost == Decimal("0.000450")


def test_claude_sonnet_cost():
    """Test Claude Sonnet 4.5 cost calculation."""
    cost = calculate_cost("claude-sonnet-4.5", input_tokens=1000, output_tokens=500)
    # Input: 1000 * $3.00/1M = $0.003
    # Output: 500 * $15.00/1M = $0.0075
    # Total: $0.0105
    assert cost == Decimal("0.010500")


def test_claude_3_sonnet_cost():
    """Test Claude 3 Sonnet cost calculation."""
    cost = calculate_cost("claude-3-sonnet-20240229", input_tokens=1000, output_tokens=500)
    # Input: 1000 * $3.00/1M = $0.003
    # Output: 500 * $15.00/1M = $0.0075
    # Total: $0.0105
    assert cost == Decimal("0.010500")


def test_o3_reasoning_tokens():
    """Test o3 model with reasoning tokens."""
    cost = calculate_cost("o3", input_tokens=1000, output_tokens=500, reasoning_tokens=2000)
    # Input: 1000 * $2.00/1M = $0.002
    # Output: 500 * $8.00/1M = $0.004
    # Reasoning: 2000 * $8.00/1M = $0.016 (billed as output)
    # Total: $0.022
    assert cost == Decimal("0.022000")


def test_o3_mini_reasoning_tokens():
    """Test o3-mini with reasoning tokens."""
    cost = calculate_cost("o3-mini", input_tokens=1000, output_tokens=500, reasoning_tokens=1000)
    # Input: 1000 * $1.10/1M = $0.0011
    # Output: 500 * $4.40/1M = $0.0022
    # Reasoning: 1000 * $4.40/1M = $0.0044 (billed as output)
    # Total: $0.0077
    assert cost == Decimal("0.007700")


def test_unknown_model_fallback():
    """Unknown models should use gpt-4o pricing."""
    cost = calculate_cost("unknown-model", input_tokens=1000, output_tokens=500)
    expected = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
    assert cost == expected


def test_zero_tokens():
    """Test with zero tokens."""
    cost = calculate_cost("gpt-4o", input_tokens=0, output_tokens=0)
    assert cost == Decimal("0.000000")


def test_large_token_count():
    """Test with large token counts."""
    cost = calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
    # Input: 1M * $2.50/1M = $2.50
    # Output: 0.5M * $10/1M = $5.00
    # Total: $7.50
    assert cost == Decimal("7.500000")


def test_decimal_precision():
    """Test that costs are rounded to 6 decimal places."""
    cost = calculate_cost("gpt-4o", input_tokens=1, output_tokens=1)
    # Input: 1 * $2.50/1M = $0.0000025
    # Output: 1 * $10/1M = $0.00001
    # Total: $0.0000125 rounded to 6 decimals = $0.000012
    assert cost == Decimal("0.000012")
    # Verify string representation has 6 decimals
    cost_str = str(cost)
    if "." in cost_str:
        decimals = len(cost_str.split(".")[1])
        assert decimals <= 6


def test_no_reasoning_tokens_default():
    """Test that reasoning tokens default to 0 if not provided."""
    cost_without = calculate_cost("o3", input_tokens=1000, output_tokens=500)
    cost_with_zero = calculate_cost("o3", input_tokens=1000, output_tokens=500, reasoning_tokens=0)
    assert cost_without == cost_with_zero
