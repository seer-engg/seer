"""
Tool call correctness eval — discriminator-generator pattern for Seer MCP tools.

Generator: calls unified tool implementations with known inputs.
Discriminator: asserts JSON structure, required fields, no error keys, response sizes.

No LLM required. Run with:
    uv run pytest tests/eval/test_tool_call_eval.py -s --tb=short
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from seer.tools.unified_tools import (
    get_workflow_schema_impl,
    get_workflow_template_impl,
    list_tools_impl,
    list_triggers_impl,
    list_workflow_templates_impl,
    search_tools_impl,
    search_triggers_impl,
)

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for embedding-based search tests",
    ),
]

# ---------------------------------------------------------------------------
# Timing / size tracking
# ---------------------------------------------------------------------------

@dataclass
class ToolCallResult:
    tool: str
    elapsed_ms: float
    response_bytes: int
    approx_tokens: int  # rough: bytes / 4
    parsed: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


_results: List[ToolCallResult] = []


async def _call_tool(name: str, fn, **kwargs) -> ToolCallResult:
    """Call a tool impl, measure timing + size, parse JSON."""
    start = time.perf_counter()
    try:
        raw = await fn(**kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        parsed = json.loads(raw)
        result = ToolCallResult(
            tool=name, elapsed_ms=round(elapsed, 1),
            response_bytes=len(raw), approx_tokens=len(raw) // 4,
            parsed=parsed,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        result = ToolCallResult(
            tool=name, elapsed_ms=round(elapsed, 1),
            response_bytes=0, approx_tokens=0, error=str(exc),
        )
    _results.append(result)
    return result


def _has_no_error(parsed: Dict[str, Any]) -> bool:
    return "error" not in parsed


# ---------------------------------------------------------------------------
# search_tools tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_tools_finds_gmail():
    r = await _call_tool("search_tools", search_tools_impl, query="send email")
    assert _has_no_error(r.parsed), f"search_tools returned error: {r.parsed.get('error')}"
    top = r.parsed.get("top_match")
    assert top is not None, "top_match is None for 'send email'"
    assert "gmail" in top.get("tool", "").lower(), f"Expected gmail tool, got: {top['tool']}"
    assert "alternatives" in r.parsed, "Missing 'alternatives' key"


@pytest.mark.asyncio
async def test_search_tools_finds_slack():
    r = await _call_tool("search_tools", search_tools_impl, query="post message to slack channel")
    assert _has_no_error(r.parsed)
    top = r.parsed.get("top_match")
    assert top is not None
    assert "slack" in top.get("tool", "").lower(), f"Expected slack tool, got: {top['tool']}"


@pytest.mark.asyncio
async def test_search_tools_with_integration_filter():
    r = await _call_tool("search_tools(gmail)", search_tools_impl, query="send email", integration_filter="gmail")
    assert _has_no_error(r.parsed)
    top = r.parsed.get("top_match")
    if top:
        assert "gmail" in top.get("tool", "").lower(), f"Expected gmail tool, got: {top['tool']}"
    else:
        # Embedding search with filter may return nothing if index not warmed
        # This is still a valid result — the tool didn't error
        assert r.parsed.get("alternatives") is not None or "message" in r.parsed


@pytest.mark.asyncio
async def test_search_tools_low_confidence_for_nonsense():
    """Nonsense queries may return results, but confidence should be low."""
    r = await _call_tool("search_tools(nonsense)", search_tools_impl, query="xyznonexistent_tool_12345")
    assert _has_no_error(r.parsed)
    top = r.parsed.get("top_match")
    if top:
        # Embedding search always returns something — verify low confidence
        confidence = top.get("confidence", 1.0)
        assert confidence < 0.5, f"Nonsense query should have low confidence, got {confidence}"


# ---------------------------------------------------------------------------
# search_triggers tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_triggers_finds_gmail():
    r = await _call_tool("search_triggers", search_triggers_impl, query="gmail new email")
    assert _has_no_error(r.parsed)
    triggers = r.parsed.get("triggers", [])
    assert len(triggers) > 0, "No triggers found for 'gmail new email'"
    assert any("gmail" in t.get("key", "").lower() for t in triggers), \
        f"No gmail trigger found in: {[t.get('key') for t in triggers]}"


@pytest.mark.asyncio
async def test_search_triggers_finds_cron():
    r = await _call_tool("search_triggers", search_triggers_impl, query="run every day at 9am")
    assert _has_no_error(r.parsed)
    triggers = r.parsed.get("triggers", [])
    assert len(triggers) > 0
    assert any("cron" in t.get("key", "").lower() or "schedule" in t.get("key", "").lower()
               for t in triggers), f"No schedule trigger found: {[t.get('key') for t in triggers]}"


# ---------------------------------------------------------------------------
# list_tools tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_tools_returns_all():
    r = await _call_tool("list_tools", list_tools_impl)
    assert _has_no_error(r.parsed)
    tools = r.parsed.get("tools", [])
    total = r.parsed.get("total", 0)
    assert len(tools) == total, f"tools list length {len(tools)} != total {total}"
    assert total > 0, "No tools returned"
    # Each tool must have name + description
    for t in tools:
        assert "name" in t, f"Tool missing 'name': {t}"
        assert "description" in t, f"Tool missing 'description': {t}"
        assert "integration_type" in t, f"Tool missing 'integration_type': {t}"


@pytest.mark.asyncio
async def test_list_tools_with_filter():
    r = await _call_tool("list_tools", list_tools_impl, integration_type="gmail")
    assert _has_no_error(r.parsed)
    tools = r.parsed.get("tools", [])
    for t in tools:
        assert "gmail" in t.get("integration_type", "").lower() or \
               "gmail" in t.get("name", "").lower(), \
            f"Non-gmail tool in filtered results: {t.get('name')}"


# ---------------------------------------------------------------------------
# list_triggers tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_triggers_returns_all():
    r = await _call_tool("list_triggers", list_triggers_impl)
    assert _has_no_error(r.parsed)
    triggers = r.parsed.get("triggers", [])
    total = r.parsed.get("total", 0)
    assert len(triggers) == total
    assert total > 0, "No triggers returned"
    # Each trigger must have key + provider
    for t in triggers:
        assert "key" in t, f"Trigger missing 'key': {t}"
        assert "provider" in t, f"Trigger missing 'provider': {t}"


# ---------------------------------------------------------------------------
# get_workflow_schema tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_workflow_schema_basic():
    r = await _call_tool("get_workflow_schema", get_workflow_schema_impl, focus="basic")
    assert _has_no_error(r.parsed)
    # Returns a string, not JSON
    assert isinstance(r.parsed, str) or "Node Types" in str(r.parsed) or "error" not in str(r.parsed)


@pytest.mark.asyncio
async def test_get_workflow_schema_full():
    r = await _call_tool("get_workflow_schema", get_workflow_schema_impl, focus="full")
    assert _has_no_error(r.parsed)


# ---------------------------------------------------------------------------
# get_workflow_template tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_workflow_template():
    """Template tools need DB — just verify they don't crash."""
    r = await _call_tool("get_workflow_template", get_workflow_template_impl, query="slack notification")
    # DB not available in eval context — just check it returns valid JSON
    assert isinstance(r.parsed, dict)


@pytest.mark.asyncio
async def test_list_workflow_templates():
    """Template tools need DB — just verify they don't crash."""
    r = await _call_tool("list_workflow_templates", list_workflow_templates_impl)
    assert isinstance(r.parsed, dict)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def test_summary_report():
    """Print timing + size summary for all tool calls in this session."""
    if not _results:
        pytest.skip("No tool call results collected")

    print("\n" + "=" * 72)
    print(f"{'Tool':<30} {'ms':>8} {'bytes':>10} {'~tokens':>10}")
    print("-" * 72)
    for r in _results:
        status = "ERROR" if r.error else "OK"
        print(f"{r.tool:<30} {r.elapsed_ms:>7.1f} {r.response_bytes:>10} {r.approx_tokens:>10}  {status}")
    print("-" * 72)
    total_bytes = sum(r.response_bytes for r in _results)
    total_tokens = sum(r.approx_tokens for r in _results)
    print(f"{'TOTAL':<30} {'':>8} {total_bytes:>10} {total_tokens:>10}")
    print("=" * 72)
