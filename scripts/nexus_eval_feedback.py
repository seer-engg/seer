"""
Analyze nexus eval results and generate product improvement recommendations.

Reads ground_truth.json + raw_sessions.json and produces:
  1. Tool description improvement suggestions
  2. New golden example candidates from accepted sessions
  3. Error recovery patterns from failed sessions
  4. Clarification question quality assessment

Run:
    uv run python scripts/nexus_eval_feedback.py
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from seer.tools.discovery_shared import (
    _embedding_index,
    async_search_tools_intent,
    async_search_triggers_intent,
)

EVAL_DIR = Path(__file__).resolve().parent.parent / "tests" / "eval" / "nexus"


def load_data() -> tuple[list[dict], list[dict]]:
    raw_path = EVAL_DIR / "raw_sessions.json"
    gt_path = EVAL_DIR / "ground_truth.json"
    if not raw_path.exists() or not gt_path.exists():
        print("Run scripts/extract_nexus_threads.py first")
        raise SystemExit(1)
    return json.loads(raw_path.read_text()), json.loads(gt_path.read_text())


async def find_tool_description_gaps(
    ground_truth: list[dict],
) -> list[dict]:
    """Find tools that users expected but search missed."""
    gaps: list[dict] = []
    for session in ground_truth:
        query = session.get("first_user_message", "")
        expected = session.get("expected_tools", [])
        if not query or not expected:
            continue

        results = await async_search_tools_intent(query, top_k=10)
        found = {r["name"] for r in results}

        for tool in expected:
            if tool not in found:
                gaps.append({
                    "tool": tool,
                    "query": query[:100],
                    "session_id": session["session_id"],
                    "intent": session["intent"],
                })

    # Deduplicate by tool
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for g in gaps:
        by_tool[g["tool"]].append(g)

    recommendations = []
    for tool, instances in by_tool.items():
        recommendations.append({
            "tool": tool,
            "miss_count": len(instances),
            "sample_queries": [i["query"] for i in instances[:3]],
            "recommendation": f"Improve description of '{tool}' to match these user queries",
        })

    return sorted(recommendations, key=lambda r: r["miss_count"], reverse=True)


def extract_golden_candidates(
    raw_sessions: list[dict],
    ground_truth: list[dict],
) -> list[dict]:
    """Accepted sessions with clean single-turn workflows → golden example candidates."""
    candidates = []
    gt_by_id = {s["session_id"]: s for s in ground_truth}

    for session in raw_sessions:
        gt = gt_by_id.get(session["session_id"])
        if not gt or gt["outcome"] != "accepted":
            continue

        # Only single-proposal sessions (clean signal)
        if session["proposal_count"] != 1:
            continue

        user_msgs = [m for m in session["messages"] if m["role"] == "user"]
        if not user_msgs:
            continue

        proposal = session["proposals"][0] if session["proposals"] else None
        if not proposal or not proposal.get("spec"):
            continue

        candidates.append({
            "session_id": session["session_id"],
            "prompt": user_msgs[0]["content"][:300],
            "intent": gt["intent"],
            "user_email": gt["user_email"],
            "tools": gt["expected_tools"],
            "triggers": gt["expected_triggers"],
            "spec": proposal["spec"],
        })

    return candidates


def extract_error_patterns(
    raw_sessions: list[dict],
    ground_truth: list[dict],
) -> list[dict]:
    """Failed sessions with error patterns → system prompt improvements."""
    gt_by_id = {s["session_id"]: s for s in ground_truth}
    patterns = []

    for session in raw_sessions:
        gt = gt_by_id.get(session["session_id"])
        if not gt:
            continue

        # Sessions where user came back to fix
        user_msgs = [m for m in session["messages"] if m["role"] == "user"]
        if len(user_msgs) < 2:
            continue

        second_msg = user_msgs[1]["content"].lower() if len(user_msgs) > 1 else ""
        is_fix = any(
            kw in second_msg
            for kw in ["fix", "error", "broken", "wrong", "not working", "400", "500"]
        )
        if not is_fix:
            continue

        patterns.append({
            "session_id": session["session_id"],
            "intent": gt["intent"],
            "first_message": user_msgs[0]["content"][:200],
            "fix_message": user_msgs[1]["content"][:200],
            "outcome": gt["outcome"],
            "error_category": gt.get("error_category"),
        })

    return patterns


def print_report(
    tool_gaps: list[dict],
    golden_candidates: list[dict],
    error_patterns: list[dict],
) -> None:
    print("\n" + "=" * 80)
    print("NEXUS EVAL → PRODUCT FEEDBACK")
    print("=" * 80)

    # Tool description gaps
    print(f"\n--- Tool Description Gaps ({len(tool_gaps)}) ---")
    if tool_gaps:
        for g in tool_gaps[:10]:
            print(f"  {g['tool']}: missed {g['miss_count']}x")
            for q in g["sample_queries"]:
                print(f"    query: \"{q[:80]}...\"")
            print(f"    → {g['recommendation']}")
    else:
        print("  No gaps found — all expected tools retrieved")

    # Golden candidates
    print(f"\n--- Golden Example Candidates ({len(golden_candidates)}) ---")
    if golden_candidates:
        for c in golden_candidates[:5]:
            print(f"  [{c['session_id']}] {c['intent']}: \"{c['prompt'][:60]}...\"")
            print(f"    tools={c['tools']} triggers={c['triggers']}")
    else:
        print("  No clean single-turn accepted sessions found")

    # Error patterns
    print(f"\n--- Error Recovery Patterns ({len(error_patterns)}) ---")
    if error_patterns:
        for p in error_patterns[:5]:
            print(f"  [{p['session_id']}] {p['intent']}")
            print(f"    first: \"{p['first_message'][:60]}...\"")
            print(f"    fix:   \"{p['fix_message'][:60]}...\"")
    else:
        print("  No multi-turn fix patterns found")

    print("\n" + "=" * 80)
    print("ACTIONS:")
    if tool_gaps:
        print(f"  1. Update tool descriptions for: {', '.join(g['tool'] for g in tool_gaps[:5])}")
    if golden_candidates:
        print(f"  2. Add {len(golden_candidates)} golden examples to golden_examples.py")
    if error_patterns:
        print(f"  3. Add {len(error_patterns)} error recovery patterns to Nexus system prompt")
    print("=" * 80 + "\n")


async def main() -> None:
    raw_sessions, ground_truth = load_data()

    print(f"Analyzing {len(raw_sessions)} sessions...")
    tool_gaps = await find_tool_description_gaps(ground_truth)
    golden_candidates = extract_golden_candidates(raw_sessions, ground_truth)
    error_patterns = extract_error_patterns(raw_sessions, ground_truth)

    print_report(tool_gaps, golden_candidates, error_patterns)

    # Save findings for later use
    findings = {
        "tool_gaps": tool_gaps,
        "golden_candidates": golden_candidates,
        "error_patterns": error_patterns,
    }
    findings_path = EVAL_DIR / "product_findings.json"
    findings_path.write_text(json.dumps(findings, indent=2, default=str))
    print(f"Findings saved to {findings_path}")


if __name__ == "__main__":
    asyncio.run(main())
