"""
BFCL v3 benchmark tests for Seer tool-calling.

Runs Berkeley Function Calling Leaderboard evals in two modes:
  Mode A — Seer tools + BFCL queries (tests our search_tools discovery)
  Mode B — BFCL native function definitions (competitive benchmark)

Requires OPENAI_API_KEY. Run with:
    uv run pytest tests/eval/test_bfcl_eval.py -s --tb=short
    uv run pytest tests/eval/test_bfcl_eval.py -s -k "test_bfcl_simple" --tb=short
"""
from __future__ import annotations

import os

import pytest

from tests.eval.bfcl.runner import run_bfcl_eval, BFCLEvalResult

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for BFCL eval tests",
    ),
]

# Quick run: limit cases per category for CI
MAX_CASES = int(os.environ.get("BFCL_MAX_CASES", "50"))

# Results storage for summary
_results: dict[str, BFCLEvalResult] = {}


@pytest.mark.asyncio
async def test_bfcl_simple_category():
    """BFCL simple category: single function, single call."""
    result = await run_bfcl_eval(mode="A", categories=["simple"], max_cases_per_category=MAX_CASES)
    _results["simple"] = result
    cat = result.categories.get("simple")
    if cat:
        print(f"\nsimple: {cat.correct}/{cat.total} ({cat.accuracy:.1f}%)")


@pytest.mark.asyncio
async def test_bfcl_multiple_category():
    """BFCL multiple category: multiple functions available, single call."""
    result = await run_bfcl_eval(mode="A", categories=["multiple"], max_cases_per_category=MAX_CASES)
    _results["multiple"] = result
    cat = result.categories.get("multiple")
    if cat:
        print(f"\nmultiple: {cat.correct}/{cat.total} ({cat.accuracy:.1f}%)")


@pytest.mark.asyncio
async def test_bfcl_parallel_category():
    """BFCL parallel category: multiple simultaneous calls."""
    result = await run_bfcl_eval(mode="A", categories=["parallel"], max_cases_per_category=MAX_CASES)
    _results["parallel"] = result
    cat = result.categories.get("parallel")
    if cat:
        print(f"\nparallel: {cat.correct}/{cat.total} ({cat.accuracy:.1f}%)")


@pytest.mark.asyncio
async def test_bfcl_parallel_multiple_category():
    """BFCL parallel_multiple: multiple functions + multiple parallel calls."""
    result = await run_bfcl_eval(mode="A", categories=["parallel_multiple"], max_cases_per_category=MAX_CASES)
    _results["parallel_multiple"] = result
    cat = result.categories.get("parallel_multiple")
    if cat:
        print(f"\nparallel_multiple: {cat.correct}/{cat.total} ({cat.accuracy:.1f}%)")


@pytest.mark.asyncio
async def test_bfcl_irrelevance_category():
    """BFCL irrelevance: queries that should NOT trigger any function call."""
    result = await run_bfcl_eval(mode="A", categories=["irrelevance"], max_cases_per_category=MAX_CASES)
    _results["irrelevance"] = result
    cat = result.categories.get("irrelevance")
    if cat:
        print(f"\nirrelevance: {cat.correct}/{cat.total} ({cat.accuracy:.1f}%)")


def test_bfcl_summary_report():
    """Print summary table across all BFCL categories run in this session."""
    if not _results:
        pytest.skip("No BFCL results collected (run category tests first)")

    print("\n" + "=" * 70)
    print("BFCL v3 BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Category':<25} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print("-" * 70)

    total_correct = 0
    total_cases = 0
    for cat_name in sorted(_results.keys()):
        result = _results[cat_name]
        for name, cat in result.categories.items():
            total_correct += cat.correct
            total_cases += cat.total
            print(f"{name:<25} {cat.correct:>10} {cat.total:>10} {cat.accuracy:>11.1f}%")

    overall = (total_correct / total_cases * 100) if total_cases > 0 else 0
    print("-" * 70)
    print(f"{'OVERALL':<25} {total_correct:>10} {total_cases:>10} {overall:>11.1f}%")
    print("=" * 70)
