"""
Nexus end-to-end agent eval — tests the full agent pipeline (LLM + search + validation).

Unlike the discriminator eval which only tests search recall, this evaluates:
  1. Did the agent call search_tools with a reasonable query?
  2. Did it produce a spec containing the expected tools?
  3. Did the spec compile (no validation errors)?

Requires OPENROUTER_API_KEY for the agent LLM.

Usage:
    from tests.eval.nexus.e2e_runner import NexusE2ERunner
    runner = NexusE2ERunner("tests/eval/nexus/ground_truth.json")
    results = await runner.run_all()
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class E2ESessionResult:
    """E2E eval result for a single session."""

    session_id: int
    intent: str
    expected_tools: List[str] = field(default_factory=list)
    expected_triggers: List[str] = field(default_factory=list)

    # Agent behavior
    search_tools_called: bool = False
    search_query: str = ""
    validate_called: bool = False

    # Spec quality
    spec_tools: List[str] = field(default_factory=list)
    spec_triggers: List[str] = field(default_factory=list)
    spec_compiled: bool = False
    validation_error: Optional[str] = None

    # Timing
    total_latency_ms: float = 0.0

    @property
    def tool_coverage(self) -> float:
        """Fraction of expected tools present in spec."""
        if not self.expected_tools:
            return 1.0
        found = sum(1 for t in self.expected_tools if t in self.spec_tools)
        return found / len(self.expected_tools)

    @property
    def trigger_coverage(self) -> float:
        """Fraction of expected triggers present in spec."""
        if not self.expected_triggers:
            return 1.0
        found = sum(1 for t in self.expected_triggers if t in self.spec_triggers)
        return found / len(self.expected_triggers)

    @property
    def pass_e2e(self) -> bool:
        """Passed if spec compiled and has >=50% tool/trigger coverage."""
        return self.spec_compiled and self.tool_coverage >= 0.5 and self.trigger_coverage >= 0.5


class NexusE2ERunner:
    """Run the full Nexus agent on eval prompts and score results."""

    def __init__(self, ground_truth_path: str | Path) -> None:
        self.ground_truth_path = Path(ground_truth_path)
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

    async def eval_session(self, session: Dict[str, Any]) -> E2ESessionResult:
        """Run the full agent on a single session's first user message."""
        result = E2ESessionResult(
            session_id=session["session_id"],
            intent=session.get("intent", "unknown"),
            expected_tools=session.get("expected_tools", []),
            expected_triggers=session.get("expected_triggers", []),
        )

        query = session.get("first_user_message", "")
        if not query or (not result.expected_tools and not result.expected_triggers):
            return result

        # Skip context-dependent sessions — agent needs existing workflow context
        if session.get("context_type") == "context_dependent":
            return result

        t0 = time.perf_counter()

        try:
            from seer.agents.nexus import create_nexus_chat_agent

            agent = await create_nexus_chat_agent(
                model="google/gemini-2.0-flash-001",
                checkpointer=None,  # no persistence for eval
            )

            # Invoke agent with the user message
            # The agent will call search_tools, validate_and_upsert_workflow internally
            agent_result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
            )

            # Extract tool calls from agent result
            messages = agent_result.get("messages", [])
            for msg in messages:
                # Check for search_tools call
                if hasattr(msg, "tool_calls"):
                    for tc in msg.tool_calls:
                        if tc.get("name") == "search_tools":
                            result.search_tools_called = True
                            result.search_query = tc.get("args", {}).get("query", "")
                        if tc.get("name") == "validate_and_upsert_workflow":
                            result.validate_called = True
                            spec = tc.get("args", {}).get("spec", {})
                            # Extract tools/triggers from spec
                            for node in spec.get("nodes", []):
                                if node.get("type") == "tool" and node.get("tool"):
                                    result.spec_tools.append(node["tool"])
                            for trigger in spec.get("triggers", []):
                                if trigger.get("key"):
                                    result.spec_triggers.append(trigger["key"])

                # Check for validate_and_upsert_workflow result (compilation check)
                if hasattr(msg, "name") and msg.name == "validate_and_upsert_workflow":
                    try:
                        content = msg.content
                        if isinstance(content, str):
                            data = json.loads(content)
                            result.spec_compiled = data.get("status") == "success"
                            if not result.spec_compiled:
                                result.validation_error = data.get("message", "")
                        elif isinstance(content, dict):
                            result.spec_compiled = content.get("status") == "success"
                    except (json.JSONDecodeError, AttributeError):
                        pass

        except Exception as e:
            result.validation_error = str(e)

        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        return result

    async def run_all(self) -> List[E2ESessionResult]:
        """Run e2e eval on all standalone sessions with expectations."""
        results: List[E2ESessionResult] = []
        for session in self.sessions:
            # Only eval standalone sessions with tools/triggers
            if session.get("context_type") == "context_dependent":
                continue
            if not session.get("expected_tools") and not session.get("expected_triggers"):
                continue
            r = await self.eval_session(session)
            results.append(r)
        return results
