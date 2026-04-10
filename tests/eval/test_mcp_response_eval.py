"""
MCP response quality eval — baseline measurements for Seer MCP reliability.

Measures: response size (bytes/tokens), latency (ms), JSON validity, structural correctness.
This creates the baseline for list_tools compaction and other optimizations.

No LLM required. Run with:
    uv run pytest tests/eval/test_mcp_response_eval.py -s --tb=short
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from seer.tools.unified_tools import (
    get_workflow_schema_impl,
    list_tools_impl,
    list_triggers_impl,
    list_workflow_templates_impl,
    search_tools_impl,
    search_triggers_impl,
)

pytestmark = pytest.mark.eval


@dataclass
class Metric:
    tool: str
    query: str
    elapsed_ms: float
    response_bytes: int
    approx_tokens: int
    json_valid: bool
    has_error: bool
    notes: str = ""


_metrics: List[Metric] = []


async def _measure(tool: str, label: str, fn, **kwargs) -> Metric:
    start = time.perf_counter()
    try:
        raw = await fn(**kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        try:
            parsed = json.loads(raw)
            json_valid = True
            has_error = "error" in parsed
        except (json.JSONDecodeError, TypeError):
            # Some tools return plain text (e.g., get_workflow_schema)
            parsed = {}
            json_valid = isinstance(raw, str) and len(raw) > 0
            has_error = False

        m = Metric(
            tool=tool, query=label, elapsed_ms=round(elapsed, 1),
            response_bytes=len(raw) if isinstance(raw, str) else 0,
            approx_tokens=len(raw) // 4 if isinstance(raw, str) else 0,
            json_valid=json_valid, has_error=has_error,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        m = Metric(
            tool=tool, query=label, elapsed_ms=round(elapsed, 1),
            response_bytes=0, approx_tokens=0, json_valid=False, has_error=True,
            notes=str(exc),
        )
    _metrics.append(m)
    return m


# ---------------------------------------------------------------------------
# Baseline: list_tools full response size
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_tools_baseline_size():
    """Measure full list_tools response — this is the bloat baseline."""
    m = await _measure("list_tools", "all", list_tools_impl)
    assert not m.has_error, f"list_tools returned error"
    print(f"\n  list_tools full: {m.response_bytes:,} bytes (~{m.approx_tokens:,} tokens) in {m.elapsed_ms:.0f}ms")

    # Parse and report tool-level sizes
    raw = await list_tools_impl()
    parsed = json.loads(raw)
    tools = parsed.get("tools", [])

    # Find top-5 fattest tools by parameter schema size
    tool_sizes = []
    for t in tools:
        params_json = json.dumps(t.get("parameters", {}))
        tool_sizes.append((t.get("name", ""), len(params_json), t.get("integration_type", "")))

    tool_sizes.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top-5 fattest tool parameter schemas:")
    for name, size, integ in tool_sizes[:5]:
        print(f"    {name:<45} {size:>8} bytes  ({integ})")

    # Report total breakdown
    total_params_bytes = sum(s for _, s, _ in tool_sizes)
    total_response = m.response_bytes
    params_pct = (total_params_bytes / total_response * 100) if total_response else 0
    print(f"\n  Total parameter schemas: {total_params_bytes:,} bytes ({params_pct:.0f}% of response)")
    print(f"  Total response: {total_response:,} bytes (~{m.approx_tokens:,} tokens)")

    # Assert response is reasonable (baseline measurement, not a strict pass/fail)
    # Current expectation: 30K-60K bytes. After compact mode: <10K bytes.
    assert m.response_bytes > 0, "list_tools returned empty response"


@pytest.mark.asyncio
async def test_list_tools_gmail_size():
    """Measure filtered list_tools response."""
    m = await _measure("list_tools", "gmail", list_tools_impl, integration_type="gmail")
    assert not m.has_error
    print(f"\n  list_tools(gmail): {m.response_bytes:,} bytes (~{m.approx_tokens:,} tokens)")


# ---------------------------------------------------------------------------
# Baseline: search_tools response sizes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_tools_response_size():
    """Measure search_tools responses for common queries."""
    queries = [
        "send email",
        "create a slack message",
        "read google sheets",
        "insert data into supabase",
        "web search",
    ]
    for q in queries:
        m = await _measure("search_tools", q, search_tools_impl, query=q)
        assert not m.has_error, f"search_tools({q!r}) returned error"
        # Verify top_match doesn't have resource_pickers (or flag it)
        raw = await search_tools_impl(query=q)
        parsed = json.loads(raw)
        top = parsed.get("top_match", {})
        if top and "resource_pickers" in top:
            m.notes = "HAS resource_pickers (bloat)"


# ---------------------------------------------------------------------------
# Baseline: list_triggers response sizes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_triggers_baseline_size():
    """Measure full list_triggers response."""
    m = await _measure("list_triggers", "all", list_triggers_impl)
    assert not m.has_error
    print(f"\n  list_triggers full: {m.response_bytes:,} bytes (~{m.approx_tokens:,} tokens)")


# ---------------------------------------------------------------------------
# Baseline: search_triggers response sizes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_triggers_response_size():
    """Measure search_triggers responses."""
    queries = ["gmail new email", "cron schedule", "supabase webhook", "slack message"]
    for q in queries:
        m = await _measure("search_triggers", q, search_triggers_impl, query=q)
        assert not m.has_error, f"search_triggers({q!r}) returned error"


# ---------------------------------------------------------------------------
# Structural correctness checks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_tools_structure():
    """Verify search_tools returns consistent structure."""
    raw = await search_tools_impl(query="send email")
    parsed = json.loads(raw)

    # Must have these keys
    for key in ("query", "top_match", "alternatives"):
        assert key in parsed, f"Missing key: {key}"

    top = parsed["top_match"]
    if top:
        for key in ("tool", "integration", "confidence", "description"):
            assert key in top, f"top_match missing key: {key}"


@pytest.mark.asyncio
async def test_list_tools_structure():
    """Verify list_tools returns consistent structure."""
    raw = await list_tools_impl()
    parsed = json.loads(raw)

    for key in ("tools", "total", "integration_filter", "available_integrations"):
        assert key in parsed, f"Missing key: {key}"

    # Verify each tool has minimal required fields
    for t in parsed["tools"]:
        for key in ("name", "description", "integration_type"):
            assert key in t, f"Tool {t.get('name', '?')} missing key: {key}"


@pytest.mark.asyncio
async def test_list_triggers_structure():
    """Verify list_triggers returns consistent structure."""
    raw = await list_triggers_impl()
    parsed = json.loads(raw)

    for key in ("triggers", "total", "provider_filter"):
        assert key in parsed, f"Missing key: {key}"

    for t in parsed["triggers"]:
        for key in ("key", "provider", "mode"):
            assert key in t, f"Trigger {t.get('key', '?')} missing key: {key}"


# ---------------------------------------------------------------------------
# Latency thresholds (not strict — these are baseline measurements)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_tools_latency():
    """Measure search_tools latency — should be fast after warm-up."""
    # Warm up embedding index
    await search_tools_impl(query="warmup")

    times = []
    for _ in range(3):
        start = time.perf_counter()
        await search_tools_impl(query="send email via gmail")
        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    print(f"\n  search_tools avg latency: {avg:.0f}ms (3 runs: {[f'{t:.0f}ms' for t in times]})")
    # Baseline measurement — not enforcing strict threshold yet


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def test_mcp_response_summary():
    """Print complete baseline metrics table."""
    if not _metrics:
        pytest.skip("No metrics collected")

    print("\n" + "=" * 90)
    print(f"{'Tool':<20} {'Query':<30} {'ms':>7} {'bytes':>10} {'~tokens':>10} {'OK':>4} {'Notes'}")
    print("-" * 90)
    for m in _metrics:
        ok = "Y" if not m.has_error and m.json_valid else "N"
        print(f"{m.tool:<20} {m.query:<30} {m.elapsed_ms:>6.0f} {m.response_bytes:>10,} {m.approx_tokens:>10,} {ok:>4} {m.notes}")
    print("-" * 90)
    total_bytes = sum(m.response_bytes for m in _metrics)
    total_tokens = sum(m.approx_tokens for m in _metrics)
    print(f"{'TOTAL':<20} {'':<30} {'':>7} {total_bytes:>10,} {total_tokens:>10,}")
    print("=" * 90)
    print(f"\n  Use these baselines to measure list_tools compaction impact.")
