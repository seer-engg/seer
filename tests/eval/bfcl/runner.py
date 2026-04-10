"""
BFCL v3 eval runner for Seer tool-calling.

Two modes:
  Mode A — Seer tools + BFCL queries: tests our search_tools discovery
  Mode B — BFCL native: tests pure function-calling against leaderboard

Usage:
    from tests.eval.bfcl.runner import run_bfcl_eval
    results = await run_bfcl_eval(mode="A", categories=["simple", "multiple"])
"""
from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.eval.bfcl.adapter import (
    compare_tool_calls,
    extract_functions,
    extract_query,
    get_acceptable_args,
    get_ground_truth_for_case,
    load_answers,
    load_bfcl_dataset,
    parse_ground_truth,
    seer_tools_to_bfcl_functions,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------

@dataclass
class BFCLCaseResult:
    test_id: str
    correct: bool
    predicted: List[Dict[str, Any]]
    ground_truth: List[Dict[str, Any]]
    detail: Dict[str, Any]
    elapsed_ms: float = 0.0
    error: str = ""


@dataclass
class BFCLCategoryResult:
    category: str
    total: int = 0
    correct: int = 0
    results: List[BFCLCaseResult] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total * 100) if self.total > 0 else 0.0


@dataclass
class BFCLEvalResult:
    mode: str
    model: str
    categories: Dict[str, BFCLCategoryResult] = field(default_factory=dict)

    @property
    def total_cases(self) -> int:
        return sum(c.total for c in self.categories.values())

    @property
    def total_correct(self) -> int:
        return sum(c.correct for c in self.categories.values())

    @property
    def overall_accuracy(self) -> float:
        return (self.total_correct / self.total_cases * 100) if self.total_cases > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "model": self.model,
            "overall_accuracy": round(self.overall_accuracy, 2),
            "total_cases": self.total_cases,
            "total_correct": self.total_correct,
            "categories": {
                name: {
                    "total": cat.total,
                    "correct": cat.correct,
                    "accuracy": round(cat.accuracy, 2),
                    "avg_ms": round(cat.total_ms / cat.total, 1) if cat.total > 0 else 0,
                }
                for name, cat in self.categories.items()
            },
        }


# ---------------------------------------------------------------------------
# Mode A: Seer tools + BFCL queries
# ---------------------------------------------------------------------------

async def _run_mode_a_case(
    test_case: Dict[str, Any],
    answers: Dict[str, Any],
    seer_functions: List[Dict[str, Any]],
) -> BFCLCaseResult:
    """Mode A: use search_tools to find the right Seer tool."""
    from seer.tools.discovery_shared import async_search_tools_intent

    query = extract_query(test_case)
    gt_calls = get_ground_truth_for_case(test_case, answers)

    start = time.perf_counter()
    try:
        search_results = await async_search_tools_intent(query, top_k=5)

        if not search_results:
            elapsed = (time.perf_counter() - start) * 1000
            return BFCLCaseResult(
                test_id=test_case.get("id", ""),
                correct=len(gt_calls) == 0,  # Correct if GT is also empty
                predicted=[],
                ground_truth=gt_calls,
                detail={"reason": "no search results"},
                elapsed_ms=elapsed,
            )

        # For Mode A: find matching Seer tool names and use GT args
        predicted = []
        for gt_call in gt_calls:
            gt_name = gt_call["name"]
            best_match = _find_best_seer_tool(gt_name, search_results)
            if best_match:
                predicted.append({
                    "name": best_match,
                    "arguments": gt_call["arguments"],
                })

        # Get acceptable args for lenient scoring
        gt_raw = test_case.get("ground_truth", [])
        if not gt_raw and test_case.get("id", "") in answers:
            gt_raw = answers[test_case["id"]]
        acceptable = get_acceptable_args(gt_raw)

        elapsed = (time.perf_counter() - start) * 1000
        is_correct, detail = compare_tool_calls(
            predicted, gt_calls,
            acceptable_args=acceptable if acceptable else None,
        )

        return BFCLCaseResult(
            test_id=test_case.get("id", ""),
            correct=is_correct,
            predicted=predicted,
            ground_truth=gt_calls,
            detail=detail,
            elapsed_ms=elapsed,
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return BFCLCaseResult(
            test_id=test_case.get("id", ""),
            correct=False,
            predicted=[],
            ground_truth=gt_calls,
            detail={"reason": str(e)},
            elapsed_ms=elapsed,
            error=str(e),
        )


def _find_best_seer_tool(bfcl_name: str, search_results: List[Dict]) -> Optional[str]:
    """Find the best matching Seer tool for a BFCL function name."""
    bfcl_lower = bfcl_name.lower()

    # Direct name match
    for r in search_results:
        if r.get("name", "").lower() == bfcl_lower:
            return r["name"]

    # Substring match
    for r in search_results:
        seer_name = r.get("name", "").lower()
        if bfcl_lower in seer_name or seer_name in bfcl_lower:
            return r["name"]

    # Token overlap
    bfcl_tokens = set(bfcl_lower.replace("_", " ").replace(".", " ").split())
    best_score = 0
    best_name = None
    for r in search_results:
        seer_tokens = set(r.get("name", "").lower().replace("_", " ").split())
        overlap = len(bfcl_tokens & seer_tokens)
        if overlap > best_score:
            best_score = overlap
            best_name = r["name"]

    return best_name


# ---------------------------------------------------------------------------
# Mode B: BFCL native (competitive benchmark via LLM)
# ---------------------------------------------------------------------------

def _normalize_params(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize BFCL parameter schemas for OpenAI compatibility.

    BFCL quirks: type="dict" (should be "object"), type="intege" (typo), etc.
    """
    if not schema:
        return {"type": "object", "properties": {}}

    schema = deepcopy(schema)

    # Fix top-level type
    if schema.get("type") in ("dict", "object"):
        schema["type"] = "object"

    # Recursively fix property types
    if "properties" in schema:
        for prop in schema["properties"].values():
            if isinstance(prop, dict):
                t = prop.get("type", "")
                # Fix BFCL typos
                if t == "intege":
                    prop["type"] = "integer"
                elif t == "dict":
                    prop["type"] = "object"
                    if "properties" not in prop:
                        prop["properties"] = {}
                elif t == "arra":
                    prop["type"] = "array"
                # Recurse into nested objects
                if prop.get("type") == "object" and "properties" in prop:
                    prop = _normalize_params(prop)
                # Fix items in arrays
                if prop.get("type") == "array" and "items" in prop:
                    if isinstance(prop["items"], dict):
                        if prop["items"].get("type") == "dict":
                            prop["items"]["type"] = "object"
                        if prop["items"].get("type") == "intege":
                            prop["items"]["type"] = "integer"

    return schema


async def _run_mode_b_case(
    test_case: Dict[str, Any],
    answers: Dict[str, Any],
    model: str = "gpt-4o",
) -> BFCLCaseResult:
    """Mode B: use LLM with BFCL's own function definitions."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    query = extract_query(test_case)
    functions = extract_functions(test_case)
    gt_calls = get_ground_truth_for_case(test_case, answers)

    start = time.perf_counter()
    try:
        tools = []
        for fn in functions:
            params = _normalize_params(fn.get("parameters", {}))
            name = fn.get("name", "").replace(".", "_")  # OpenAI rejects dots in names
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": params,
                },
            })

        from dotenv import load_dotenv
        load_dotenv()
        from seer.config import config
        api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY")
        llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
        messages = [
            SystemMessage(content=(
                "You are a helpful assistant with access to the following functions. "
                "Use them when appropriate by calling them with the correct arguments. "
                "If no function is relevant, respond without calling any function."
            )),
            HumanMessage(content=query),
        ]

        response = await llm.ainvoke(messages, tools=tools)

        predicted = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                predicted.append({"name": name, "arguments": args})

        # Get acceptable args for lenient scoring
        gt_raw = test_case.get("ground_truth", [])
        if not gt_raw and test_case.get("id", "") in answers:
            gt_raw = answers[test_case["id"]]
        acceptable = get_acceptable_args(gt_raw)

        # Normalize GT names (dots → underscores) to match tool names we sent to LLM
        normalized_gt = [
            {"name": gt["name"].replace(".", "_"), "arguments": gt["arguments"]}
            for gt in gt_calls
        ]
        normalized_acceptable = [
            {"name": acc["name"].replace(".", "_"), "acceptable_args": acc["acceptable_args"]}
            for acc in (acceptable or [])
        ] if acceptable else None

        elapsed = (time.perf_counter() - start) * 1000
        is_correct, detail = compare_tool_calls(
            predicted, normalized_gt,
            acceptable_args=normalized_acceptable,
        )

        return BFCLCaseResult(
            test_id=test_case.get("id", ""),
            correct=is_correct,
            predicted=predicted,
            ground_truth=gt_calls,
            detail=detail,
            elapsed_ms=elapsed,
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return BFCLCaseResult(
            test_id=test_case.get("id", ""),
            correct=False,
            predicted=[],
            ground_truth=gt_calls,
            detail={"reason": str(e)},
            elapsed_ms=elapsed,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

CATEGORIES = [
    "simple", "multiple", "parallel", "parallel_multiple",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "multi_turn_base", "multi_turn_composite", "multi_turn_long_context",
    "multi_turn_miss_param",
    "irrelevance",
    "java", "javascript", "sql", "rest",
]


def ensure_data_downloaded() -> bool:
    """Check if BFCL data exists, download if not."""
    json_files = list(DATA_DIR.glob("*.json"))
    if len(json_files) >= 10:
        return True

    print("BFCL data not found. Downloading...")
    import subprocess
    script = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "download_bfcl.py"
    result = subprocess.run(["python", str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Download failed: {result.stderr}")
        return False
    return True


async def run_bfcl_eval(
    mode: str = "A",
    categories: Optional[List[str]] = None,
    model: str = "gpt-4o",
    max_cases_per_category: Optional[int] = None,
) -> BFCLEvalResult:
    """Run BFCL eval.

    Args:
        mode: "A" for Seer tools, "B" for BFCL native
        categories: list of category names (None = all available)
        model: LLM model for Mode B
        max_cases_per_category: limit cases per category
    """
    if not ensure_data_downloaded():
        raise RuntimeError("BFCL data not available. Run: python scripts/download_bfcl.py")

    seer_functions = []
    if mode == "A":
        from seer.tools.registry import get_tools_by_integration
        all_tools = get_tools_by_integration()
        seer_functions = seer_tools_to_bfcl_functions(all_tools)

    cats_to_run = categories or CATEGORIES
    result = BFCLEvalResult(mode=mode, model=model)

    for cat_name in cats_to_run:
        filepath = DATA_DIR / f"{cat_name}.json"
        if not filepath.exists():
            print(f"  SKIP {cat_name} (file not found)")
            continue

        cases = load_bfcl_dataset(str(filepath))
        if max_cases_per_category:
            cases = cases[:max_cases_per_category]

        # Load answers for this category
        answers = load_answers(cat_name)

        cat_result = BFCLCategoryResult(category=cat_name)
        print(f"\n  {cat_name}: {len(cases)} cases...", end="", flush=True)

        for i, case in enumerate(cases):
            if mode == "A":
                case_result = await _run_mode_a_case(case, answers, seer_functions)
            else:
                case_result = await _run_mode_b_case(case, answers, model=model)

            cat_result.results.append(case_result)
            cat_result.total += 1
            cat_result.total_ms += case_result.elapsed_ms
            if case_result.correct:
                cat_result.correct += 1

            if (i + 1) % 50 == 0:
                print(f" {i+1}", end="", flush=True)

        result.categories[cat_name] = cat_result
        print(f" → {cat_result.correct}/{cat_result.total} ({cat_result.accuracy:.1f}%)")

    return result
