"""
End-to-end Nexus agent eval — tests the full agent pipeline with a real LLM.

This is a scaffold with a single test case. It requires OPENROUTER_API_KEY
and is slow (calls an external LLM). Mark it @pytest.mark.eval.

Run:
    uv run pytest tests/eval/test_nexus_e2e_eval.py -s --tb=short
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.eval.nexus.e2e_runner import NexusE2ERunner

EVAL_DIR = Path(__file__).resolve().parent / "nexus"
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY required for agent LLM",
    ),
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for embedding search",
    ),
    pytest.mark.skipif(
        not GROUND_TRUTH_PATH.exists(),
        reason="ground_truth.json not found — run scripts/extract_nexus_threads.py",
    ),
]


@pytest.mark.asyncio
async def test_nexus_e2e_single_session():
    """
    E2E eval on a single session — scaffold for the full agent pipeline.

    This tests that the agent can produce a compilable spec for a real user
    prompt, not just that search finds the right tools. Expect varying quality
    from the cheap model — failures here indicate areas for agent improvement.
    """
    runner = NexusE2ERunner(GROUND_TRUTH_PATH)
    results = await runner.run_all()

    if not results:
        pytest.skip("No standalone sessions with expectations in ground truth")

    # Run the first session as a trial
    result = results[0]

    print(f"\n{'=' * 60}")
    print(f"E2E Eval: Session {result.session_id} ({result.intent})")
    print(f"{'=' * 60}")
    print(f"  Expected tools:    {result.expected_tools}")
    print(f"  Expected triggers: {result.expected_triggers}")
    print(f"  Search called:     {result.search_tools_called}")
    print(f"  Validate called:   {result.validate_called}")
    print(f"  Spec tools:        {result.spec_tools}")
    print(f"  Spec triggers:     {result.spec_triggers}")
    print(f"  Spec compiled:     {result.spec_compiled}")
    print(f"  Tool coverage:     {result.tool_coverage:.0%}")
    print(f"  Trigger coverage:  {result.trigger_coverage:.0%}")
    print(f"  Latency:           {result.total_latency_ms:.0f}ms")
    if result.validation_error:
        print(f"  Error:             {result.validation_error[:200]}")
    print(f"  Pass:              {result.pass_e2e}")
    print(f"{'=' * 60}")

    # Informational only — don't assert pass/fail yet.
    # This is a scaffold to surface agent quality issues.
    # Uncomment below once baseline is established:
    # assert result.search_tools_called, "Agent did not call search_tools"
    # assert result.spec_compiled, f"Spec did not compile: {result.validation_error}"
