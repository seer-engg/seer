"""
Production-based Nexus eval — real user prompts scored against ground truth.

Runs three eval dimensions in two modes:
  Direct mode (default):
    1. Tool selection recall (search_tools vs expected)
    2. Trigger selection recall (search_triggers vs expected)
    3. Summary report with per-category breakdown
  MCP mode:
    4. Tool selection via MCP server path (search_tools_impl)
    5. Trigger selection via MCP server path (search_triggers_impl)

Requires OPENAI_API_KEY (for embedding search) and ground_truth.json
(populated by scripts/extract_nexus_threads.py).

Run:
    uv run pytest tests/eval/test_nexus_prod_eval.py -s --tb=short
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest

from seer.tools.discovery_shared import _embedding_index

from tests.eval.nexus.runner import NexusEvalRunner, SessionResult

EVAL_DIR = Path(__file__).resolve().parent / "nexus"
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for embedding search",
    ),
    pytest.mark.skipif(
        not GROUND_TRUTH_PATH.exists(),
        reason="ground_truth.json not found — run scripts/extract_nexus_threads.py",
    ),
]

# Minimum pass rates
MIN_TOOL_RECALL = 0.5
MIN_TRIGGER_RECALL = 0.5
MIN_DISCRIMINATOR_PASS_RATE = 0.5


@pytest.fixture(autouse=True)
def _reset_index():
    _embedding_index.invalidate()
    yield
    _embedding_index.invalidate()


def _load_runner(eval_mode: str = "direct") -> NexusEvalRunner:
    return NexusEvalRunner(GROUND_TRUTH_PATH, eval_mode=eval_mode)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Individual session scoring
# ---------------------------------------------------------------------------


async def _score_sessions(eval_mode: str = "direct") -> List[SessionResult]:
    runner = _load_runner(eval_mode)
    report = await runner.run_all()
    mode_label = f" [{eval_mode}]" if eval_mode != "direct" else ""
    print(f"\n{report.summary_table()}{mode_label}")
    return report.results


@pytest.mark.asyncio
async def test_nexus_tool_selection():
    """Tool search recall on real user prompts (standalone sessions only)."""
    results = await _score_sessions()

    # Only score standalone sessions — context_dependent sessions can't be fairly
    # measured from the first message alone (tools are in existing workflow)
    standalone = [r for r in results if r.context_type == "standalone"]
    sessions_with_tools = [r for r in standalone if r.tools_expected]
    assert sessions_with_tools, "No standalone sessions with expected tools in ground truth"

    fails = []
    for r in sessions_with_tools:
        if r.tools_recall < MIN_TOOL_RECALL:
            fails.append(
                f"  session {r.session_id} ({r.intent}): "
                f"tool recall {r.tools_recall:.0%}, expected {r.tools_expected}, "
                f"found {r.tools_found[:5]}"
            )

    avg_recall = sum(r.tools_recall for r in sessions_with_tools) / len(sessions_with_tools)
    print(f"\nTool recall: {avg_recall:.0%} across {len(sessions_with_tools)} standalone sessions")

    if fails:
        pytest.fail(
            f"Tool recall below {MIN_TOOL_RECALL:.0%} in {len(fails)} sessions:\n"
            + "\n".join(fails)
        )


@pytest.mark.asyncio
async def test_nexus_trigger_selection():
    """Trigger search recall on real user prompts (standalone sessions only)."""
    results = await _score_sessions()

    standalone = [r for r in results if r.context_type == "standalone"]
    sessions_with_triggers = [r for r in standalone if r.triggers_expected]
    assert sessions_with_triggers, "No standalone sessions with expected triggers in ground truth"

    fails = []
    for r in sessions_with_triggers:
        if r.triggers_recall < MIN_TRIGGER_RECALL:
            fails.append(
                f"  session {r.session_id} ({r.intent}): "
                f"trigger recall {r.triggers_recall:.0%}, expected {r.triggers_expected}, "
                f"found {r.triggers_found[:5]}"
            )

    avg_recall = sum(r.triggers_recall for r in sessions_with_triggers) / len(sessions_with_triggers)
    print(f"\nTrigger recall: {avg_recall:.0%} across {len(sessions_with_triggers)} standalone sessions")

    if fails:
        pytest.fail(
            f"Trigger recall below {MIN_TRIGGER_RECALL:.0%} in {len(fails)} sessions:\n"
            + "\n".join(fails)
        )


@pytest.mark.asyncio
async def test_nexus_discriminator_pass_rate():
    """Combined discriminator pass rate must meet threshold (standalone sessions only)."""
    results = await _score_sessions()

    standalone = [r for r in results if r.context_type == "standalone"]
    sessions_with_expectations = [
        r for r in standalone if r.tools_expected or r.triggers_expected
    ]
    assert sessions_with_expectations, "No standalone sessions with expectations in ground truth"

    pass_count = sum(1 for r in sessions_with_expectations if r.discriminator_pass)
    pass_rate = pass_count / len(sessions_with_expectations)

    pass_label = f"{pass_count}/{len(sessions_with_expectations)}"
    print(f"\nDiscriminator pass rate: {pass_rate:.0%} ({pass_label}) standalone sessions")

    assert pass_rate >= MIN_DISCRIMINATOR_PASS_RATE, (
        f"Discriminator pass rate {pass_rate:.0%} below {MIN_DISCRIMINATOR_PASS_RATE:.0%}"
    )


@pytest.mark.asyncio
async def test_nexus_summary_report():
    """Print per-category accuracy table (always passes, informational)."""
    runner = _load_runner()
    report = await runner.run_all()
    print(report.summary_table())

    # Always passes — this is for dashboard output
    assert report.total_sessions > 0, "No sessions in ground truth"


# ---------------------------------------------------------------------------
# MCP path eval — tests the actual MCP server functions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nexus_mcp_tool_selection():
    """Tool search recall via MCP server path (search_tools_impl)."""
    results = await _score_sessions(eval_mode="mcp")

    standalone = [r for r in results if r.context_type == "standalone"]
    sessions_with_tools = [r for r in standalone if r.tools_expected]
    assert sessions_with_tools, "No standalone sessions with expected tools in ground truth"

    fails = []
    for r in sessions_with_tools:
        if r.tools_recall < MIN_TOOL_RECALL:
            fails.append(
                f"  session {r.session_id} ({r.intent}) [MCP]: "
                f"tool recall {r.tools_recall:.0%}, expected {r.tools_expected}, "
                f"found {r.tools_found[:5]}"
            )

    avg_recall = sum(r.tools_recall for r in sessions_with_tools) / len(sessions_with_tools)
    print(f"\n[MCP] Tool recall: {avg_recall:.0%} across {len(sessions_with_tools)} standalone sessions")

    if fails:
        pytest.fail(
            f"MCP tool recall below {MIN_TOOL_RECALL:.0%} in {len(fails)} sessions:\n"
            + "\n".join(fails)
        )


@pytest.mark.asyncio
async def test_nexus_mcp_trigger_selection():
    """Trigger search recall via MCP server path (search_triggers_impl)."""
    results = await _score_sessions(eval_mode="mcp")

    standalone = [r for r in results if r.context_type == "standalone"]
    sessions_with_triggers = [r for r in standalone if r.triggers_expected]
    assert sessions_with_triggers, "No standalone sessions with expected triggers in ground truth"

    fails = []
    for r in sessions_with_triggers:
        if r.triggers_recall < MIN_TRIGGER_RECALL:
            fails.append(
                f"  session {r.session_id} ({r.intent}) [MCP]: "
                f"trigger recall {r.triggers_recall:.0%}, expected {r.triggers_expected}, "
                f"found {r.triggers_found[:5]}"
            )

    avg_recall = sum(r.triggers_recall for r in sessions_with_triggers) / len(sessions_with_triggers)
    print(f"\n[MCP] Trigger recall: {avg_recall:.0%} across {len(sessions_with_triggers)} standalone sessions")

    if fails:
        pytest.fail(
            f"MCP trigger recall below {MIN_TRIGGER_RECALL:.0%} in {len(fails)} sessions:\n"
            + "\n".join(fails)
        )
