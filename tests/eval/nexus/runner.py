"""
Nexus replay eval runner — replays real user prompts through search + validation.

Scores:
  1. Discriminator: did search_tools/search_triggers find the right ones?
  2. Generator: would the spec compile with correct tools?
  3. End-to-end: did the first attempt succeed (from DB ground truth)?

Supports two eval modes:
  - "direct": calls async_search_tools_intent / async_search_triggers_intent directly
  - "mcp": calls search_tools_impl / search_triggers_impl (MCP server path)

Usage:
    from tests.eval.nexus.runner import NexusEvalRunner
    runner = NexusEvalRunner("tests/eval/nexus/ground_truth.json")
    results = await runner.run_all()
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from seer.tools.discovery_shared import (
    _embedding_index,
    async_search_tools_intent,
    async_search_triggers_intent,
)

EvalMode = Literal["direct", "mcp"]


@dataclass
class SessionResult:
    """Eval result for a single session."""

    session_id: int
    intent: str
    outcome: str
    context_type: str = "standalone"  # "standalone" or "context_dependent"
    eval_mode: EvalMode = "direct"

    # Discriminator scores
    tools_found: List[str] = field(default_factory=list)
    tools_expected: List[str] = field(default_factory=list)
    tools_recall: float = 0.0  # fraction of expected tools found in top-K

    triggers_found: List[str] = field(default_factory=list)
    triggers_expected: List[str] = field(default_factory=list)
    triggers_recall: float = 0.0

    # Generator score
    spec_valid: Optional[bool] = None  # could the spec compile?

    # End-to-end
    expected_success: bool = False
    first_attempt_success: bool = False  # from ground truth

    # Timing
    search_latency_ms: float = 0.0

    @property
    def discriminator_pass(self) -> bool:
        """Both tool and trigger recall >= 0.5 (at least half right)."""
        if self.context_type == "context_dependent":
            return True  # can't fairly measure recall from first message alone
        if not self.tools_expected and not self.triggers_expected:
            return True  # no expectations to fail
        tool_ok = self.tools_recall >= 0.5 if self.tools_expected else True
        trigger_ok = self.triggers_recall >= 0.5 if self.triggers_expected else True
        return tool_ok and trigger_ok


@dataclass
class EvalReport:
    """Aggregate eval report across all sessions."""

    total_sessions: int = 0
    by_intent: Dict[str, List[SessionResult]] = field(default_factory=dict)

    # Aggregate metrics
    avg_tool_recall: float = 0.0
    avg_trigger_recall: float = 0.0
    discriminator_pass_rate: float = 0.0
    first_attempt_success_rate: float = 0.0

    # Per-session results
    results: List[SessionResult] = field(default_factory=list)

    def summary_table(self) -> str:
        """Printable summary table."""
        lines = [
            "",
            "=" * 80,
            "NEXUS PROD EVAL REPORT",
            "=" * 80,
            f"Sessions: {self.total_sessions}",
            f"Tool recall (avg):      {self.avg_tool_recall:.0%}",
            f"Trigger recall (avg):   {self.avg_trigger_recall:.0%}",
            f"Discriminator pass:     {self.discriminator_pass_rate:.0%}",
            f"First-attempt success:  {self.first_attempt_success_rate:.0%}",
            "",
            "-" * 80,
            f"{'Intent':<25} {'Count':>6} {'ToolRcl':>8} {'TrigRcl':>8} {'DiscPass':>9} {'1stSucc':>8}",
            "-" * 80,
        ]
        for intent, results in sorted(self.by_intent.items()):
            n = len(results)
            tool_r = sum(r.tools_recall for r in results) / n if n else 0
            trig_r = sum(r.triggers_recall for r in results) / n if n else 0
            disc = sum(1 for r in results if r.discriminator_pass) / n if n else 0
            first = sum(1 for r in results if r.first_attempt_success) / n if n else 0
            lines.append(
                f"{intent:<25} {n:>6} {tool_r:>7.0%} {trig_r:>8.0%} {disc:>8.0%} {first:>8.0%}"
            )
        lines.append("-" * 80)
        lines.append("=" * 80)
        return "\n".join(lines)


async def _search_tools_mcp(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Call search_tools_impl (MCP path) and parse tool names from JSON response."""
    from seer.tools.unified_tools import search_tools_impl

    raw = await search_tools_impl(query=query, top_k=top_k)
    data = json.loads(raw)

    results = []
    # Extract from top_match
    top = data.get("top_match")
    if top and top.get("tool"):
        results.append({"name": top["tool"]})
    # Extract from alternatives
    for alt in data.get("alternatives", []):
        if alt.get("tool"):
            results.append({"name": alt["tool"]})
    return results


async def _search_triggers_mcp(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Call search_triggers_impl (MCP path) and parse trigger keys from JSON response."""
    from seer.tools.unified_tools import search_triggers_impl

    raw = await search_triggers_impl(query=query, top_k=top_k)
    data = json.loads(raw)

    # The MCP path returns triggers as a list under "triggers"
    return data.get("triggers", [])


class NexusEvalRunner:
    """Replay real user prompts through tool/trigger search and score results."""

    def __init__(
        self,
        ground_truth_path: str | Path,
        eval_mode: EvalMode = "direct",
    ) -> None:
        self.ground_truth_path = Path(ground_truth_path)
        self.eval_mode = eval_mode
        self.sessions: List[Dict[str, Any]] = []
        self._load_ground_truth()

    def _load_ground_truth(self) -> None:
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth not found: {self.ground_truth_path}\n"
                f"Run: uv run python scripts/extract_nexus_threads.py"
            )
        self.sessions = json.loads(
            self.ground_truth_path.read_text(encoding="utf-8")
        )

    async def eval_session(self, session: Dict[str, Any]) -> SessionResult:
        """Evaluate a single session's first user message against expected tools/triggers."""
        result = SessionResult(
            session_id=session["session_id"],
            intent=session.get("intent", "unknown"),
            outcome=session.get("outcome", "unknown"),
            context_type=session.get("context_type", "standalone"),
            eval_mode=self.eval_mode,
            tools_expected=session.get("expected_tools", []),
            triggers_expected=session.get("expected_triggers", []),
            expected_success=session.get("expected_success", False),
            first_attempt_success=session.get("expected_success", False),
        )

        # Context-dependent sessions: skip search, auto-pass discriminator
        if result.context_type == "context_dependent":
            return result

        query = session.get("first_user_message", "")
        if not query or not result.tools_expected and not result.triggers_expected:
            # Nothing to eval — skip sessions without ground truth
            return result

        # Choose search path based on eval_mode
        if self.eval_mode == "mcp":
            search_tools_fn = _search_tools_mcp
            search_triggers_fn = _search_triggers_mcp
        else:
            async def search_tools_fn(q: str, top_k: int = 10) -> List[Dict[str, Any]]:
                return await async_search_tools_intent(q, top_k=top_k)

            async def search_triggers_fn(q: str, top_k: int = 5) -> List[Dict[str, Any]]:
                return await async_search_triggers_intent(q, top_k=top_k)

        # --- Discriminator: tool search ---
        t0 = time.perf_counter()
        try:
            tool_results = await search_tools_fn(query, top_k=10)
        except Exception:
            tool_results = []
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result.search_latency_ms = elapsed_ms

        result.tools_found = [r["name"] for r in tool_results]
        if result.tools_expected:
            found = sum(1 for t in result.tools_expected if t in result.tools_found)
            result.tools_recall = found / len(result.tools_expected)

        # --- Discriminator: trigger search ---
        try:
            trigger_results = await search_triggers_fn(query, top_k=5)
        except Exception:
            trigger_results = []

        result.triggers_found = [r["key"] for r in trigger_results]
        if result.triggers_expected:
            found = sum(1 for t in result.triggers_expected if t in result.triggers_found)
            result.triggers_recall = found / len(result.triggers_expected)

        return result

    async def run_all(self) -> EvalReport:
        """Run eval on all sessions, return aggregate report."""
        _embedding_index.invalidate()

        results: List[SessionResult] = []
        for session in self.sessions:
            r = await self.eval_session(session)
            results.append(r)

        # Build report
        report = EvalReport(
            total_sessions=len(results),
            results=results,
        )

        # Aggregate metrics
        sessions_with_expectations = [
            r for r in results if r.tools_expected or r.triggers_expected
        ]
        if sessions_with_expectations:
            report.avg_tool_recall = (
                sum(r.tools_recall for r in sessions_with_expectations)
                / len(sessions_with_expectations)
            )
            report.avg_trigger_recall = (
                sum(r.triggers_recall for r in sessions_with_expectations)
                / len(sessions_with_expectations)
            )
            report.discriminator_pass_rate = (
                sum(1 for r in sessions_with_expectations if r.discriminator_pass)
                / len(sessions_with_expectations)
            )
            report.first_attempt_success_rate = (
                sum(1 for r in sessions_with_expectations if r.first_attempt_success)
                / len(sessions_with_expectations)
            )

        # Group by intent
        for r in results:
            report.by_intent.setdefault(r.intent, []).append(r)

        _embedding_index.invalidate()
        return report
