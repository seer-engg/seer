#!/usr/bin/env python3
"""
Seer Eval Runner — one command to measure everything.

Usage:
    python scripts/run_evals.py              # run all evals, show dashboard
    python scripts/run_evals.py --baseline   # run + save baseline snapshot
    python scripts/run_evals.py --compare    # diff against last baseline
    python scripts/run_evals.py --only tool  # run only tool call eval
    python scripts/run_evals.py --only mcp   # run only MCP response eval
    python scripts/run_evals.py --only retrieval  # run only retrieval eval (needs OPENAI_API_KEY)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).resolve().parent.parent / "tests" / "eval"
BASELINE_DIR = EVAL_DIR / "baselines"
BASELINE_DIR.mkdir(exist_ok=True)

EVAL_SUITES = {
    "tool": {
        "file": "test_tool_call_eval.py",
        "desc": "Tool call correctness (no LLM)",
        "needs_key": False,
    },
    "mcp": {
        "file": "test_mcp_response_eval.py",
        "desc": "MCP response quality + sizes (no LLM)",
        "needs_key": False,
    },
    "retrieval": {
        "file": "test_retrieval_eval.py",
        "desc": "Tool/trigger retrieval accuracy (needs OPENAI_API_KEY)",
        "needs_key": True,
    },
}

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _green(s: str) -> str:
    return f"{GREEN}{s}{RESET}"


def _red(s: str) -> str:
    return f"{RED}{s}{RESET}"


def _yellow(s: str) -> str:
    return f"{YELLOW}{s}{RESET}"


def _cyan(s: str) -> str:
    return f"{CYAN}{s}{RESET}"


def _bold(s: str) -> str:
    return f"{BOLD}{s}{RESET}"


def _dim(s: str) -> str:
    return f"{DIM}{s}{RESET}"


# ---------------------------------------------------------------------------
# Run pytest
# ---------------------------------------------------------------------------

def _run_suite(suite_key: str) -> Dict[str, Any]:
    """Run one eval suite, return parsed results."""
    suite = EVAL_SUITES[suite_key]
    filepath = EVAL_DIR / suite["file"]

    # Load .env if it exists (for OPENAI_API_KEY etc.)
    env = os.environ.copy()
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key not in env:
                env[key] = val

    if suite["needs_key"] and not env.get("OPENAI_API_KEY"):
        return {
            "suite": suite_key,
            "status": "skipped",
            "reason": "OPENAI_API_KEY not set",
            "passed": 0,
            "failed": 0,
            "duration_ms": 0,
            "output": "",
        }

    start = time.perf_counter()
    result = subprocess.run(
        ["uv", "run", "pytest", str(filepath), "-s", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
    )
    elapsed = (time.perf_counter() - start) * 1000

    output = result.stdout + result.stderr
    # Parse passed/failed from pytest summary line
    passed = 0
    failed = 0
    for line in output.split("\n"):
        stripped = line.strip()
        # Match patterns like "14 passed, 2 warnings" or "3 failed, 10 passed"
        if "passed" in stripped and not stripped.startswith("="):
            parts = stripped.replace("=", "").split()
            for i, p in enumerate(parts):
                if p == "passed" or p == "passed,":
                    try:
                        passed = int(parts[i - 1].rstrip(","))
                    except (ValueError, IndexError):
                        pass
                if p == "failed" or p == "failed,":
                    try:
                        failed = int(parts[i - 1].rstrip(","))
                    except (ValueError, IndexError):
                        pass

    return {
        "suite": suite_key,
        "status": "failed" if failed > 0 or result.returncode != 0 else "passed",
        "passed": passed,
        "failed": failed,
        "duration_ms": round(elapsed),
        "output": output,
        "returncode": result.returncode,
    }


# ---------------------------------------------------------------------------
# Parse metrics from eval output
# ---------------------------------------------------------------------------

def _extract_metrics(output: str) -> Dict[str, Any]:
    """Extract key metrics from eval stdout."""
    metrics = {}

    # list_tools size
    for line in output.split("\n"):
        if "list_tools full:" in line:
            # Format: "  list_tools full: 175,982 bytes (~43,995 tokens) in 11ms"
            try:
                metrics["list_tools_bytes"] = int(line.split("bytes")[0].split(":")[-1].strip().replace(",", ""))
            except (ValueError, IndexError):
                pass
            try:
                tokens_part = line.split("(~")[1].split("tokens")[0].strip().replace(",", "")
                metrics["list_tools_tokens"] = int(tokens_part)
            except (ValueError, IndexError):
                pass

        if "list_triggers full:" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if "bytes" in p:
                    try:
                        metrics["list_triggers_bytes"] = int(parts[i - 1].replace(",", ""))
                    except (ValueError, IndexError):
                        pass

        if "Overall:" in line and "%" in line:
            try:
                metrics["retrieval_pass_rate"] = float(line.split("(")[1].split("%")[0])
            except (IndexError, ValueError):
                pass

        if "Golden examples:" in line and "%" in line:
            try:
                metrics["golden_pass_rate"] = float(line.split("(")[1].split("%")[0])
            except (IndexError, ValueError):
                pass

        if "search_tools avg latency:" in line:
            try:
                metrics["search_latency_ms"] = float(line.split("avg latency:")[1].split("ms")[0].strip())
            except (IndexError, ValueError):
                pass

    return metrics


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def _print_dashboard(results: List[Dict[str, Any]]) -> None:
    """Print a clean eval dashboard."""
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_skipped = sum(1 for r in results if r["status"] == "skipped")
    all_passed = total_failed == 0 and total_skipped == 0

    print()
    print(_bold("=" * 70))
    print(_bold("  SEER EVAL DASHBOARD"))
    print(_bold("=" * 70))

    # Suite results
    for r in results:
        suite = EVAL_SUITES[r["suite"]]
        status_str = {
            "passed": _green("PASS"),
            "failed": _red("FAIL"),
            "skipped": _yellow("SKIP"),
        }.get(r["status"], r["status"])

        desc = suite["desc"]
        duration = f"{r['duration_ms'] / 1000:.1f}s"
        count = f"{r['passed']}/{r['passed'] + r['failed']}"
        reason = f" ({r.get('reason', '')})" if r["status"] == "skipped" else ""

        print(f"  {status_str}  {r['suite']:<12} {count:<8} {duration:<8} {desc}{reason}")

    # Metrics
    all_output = "\n".join(r.get("output", "") for r in results)
    metrics = _extract_metrics(all_output)
    if metrics:
        print()
        print(_bold("  KEY METRICS"))
        print("  " + "-" * 66)
        if "list_tools_bytes" in metrics:
            tokens = metrics.get("list_tools_tokens", "?")
            tokens_str = f"~{int(tokens):,}" if isinstance(tokens, (int, float)) else f"~{tokens}"
            print(f"  list_tools response    {metrics['list_tools_bytes']:>10,} bytes  ({tokens_str} tokens)")
        if "list_triggers_bytes" in metrics:
            print(f"  list_triggers response {metrics['list_triggers_bytes']:>10,} bytes")
        if "search_latency_ms" in metrics:
            print(f"  search_tools latency   {metrics['search_latency_ms']:>10.0f} ms")
        if "retrieval_pass_rate" in metrics:
            rate = metrics["retrieval_pass_rate"]
            color = _green if rate >= 95 else _red
            print(f"  retrieval accuracy     {color(f'{rate:.0f}%'):>13}")
        if "golden_pass_rate" in metrics:
            rate = metrics["golden_pass_rate"]
            color = _green if rate == 100 else _red
            print(f"  golden examples        {color(f'{rate:.0f}%'):>13}")

    # Summary
    print()
    print("  " + "-" * 66)
    if all_passed:
        print(f"  {_green(_bold('ALL PASS'))}  {total_passed} tests in {sum(r['duration_ms'] for r in results) / 1000:.1f}s")
    else:
        parts = [f"{total_passed} passed"]
        if total_failed:
            parts.append(_red(f"{total_failed} failed"))
        if total_skipped:
            parts.append(_yellow(f"{total_skipped} skipped"))
        print(f"  {_bold('RESULTS')}: {', '.join(parts)}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def _save_baseline(results: List[Dict[str, Any]]) -> Path:
    """Save baseline snapshot."""
    all_output = "\n".join(r.get("output", "") for r in results)
    metrics = _extract_metrics(all_output)
    baseline = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suites": {
            r["suite"]: {
                "passed": r["passed"],
                "failed": r["failed"],
                "status": r["status"],
                "duration_ms": r["duration_ms"],
            }
            for r in results
        },
        "metrics": metrics,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = BASELINE_DIR / f"{ts}.json"
    path.write_text(json.dumps(baseline, indent=2))
    return path


def _load_latest_baseline() -> Optional[Dict[str, Any]]:
    """Load the most recent baseline."""
    baselines = sorted(BASELINE_DIR.glob("*.json"))
    if not baselines:
        return None
    return json.loads(baselines[-1].read_text())


def _print_comparison(results: List[Dict[str, Any]]) -> None:
    """Compare current run against last baseline."""
    baseline = _load_latest_baseline()
    if not baseline:
        print(_yellow("\n  No baseline found. Run with --baseline first.\n"))
        return

    all_output = "\n".join(r.get("output", "") for r in results)
    current_metrics = _extract_metrics(all_output)
    prev_metrics = baseline.get("metrics", {})
    prev_time = baseline.get("timestamp", "unknown")

    print()
    print(_bold("=" * 70))
    print(_bold(f"  BASELINE COMPARISON"))
    print(_dim(f"  vs baseline from {prev_time}"))
    print("=" * 70)

    all_keys = sorted(set(list(current_metrics.keys()) + list(prev_metrics.keys())))
    has_regression = False

    for key in all_keys:
        curr = current_metrics.get(key)
        prev = prev_metrics.get(key)
        if curr is None or prev is None:
            continue

        # Determine if higher is better
        higher_better = key in ("retrieval_pass_rate", "golden_pass_rate")
        lower_better = key in ("list_tools_bytes", "list_tools_tokens", "list_triggers_bytes",
                               "search_latency_ms")

        diff = curr - prev
        if higher_better:
            sign = "+" if diff > 0 else ""
            color = _green if diff >= 0 else _red
        elif lower_better:
            sign = "+" if diff > 0 else ""
            color = _green if diff <= 0 else _red
        else:
            sign = "+" if diff > 0 else ""
            color = _yellow

        if (higher_better and diff < 0) or (lower_better and diff > 0):
            has_regression = True

        pct = f"({diff / prev * 100:+.1f}%)" if prev != 0 else ""
        print(f"  {key:<24} {prev:>10} → {curr:>10}  {color(f'{sign}{diff}')} {pct}")

    print("=" * 70)
    if has_regression:
        print(f"  {_red(_bold('REGRESSIONS DETECTED'))}")
    else:
        print(f"  {_green(_bold('NO REGRESSIONS'))}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    only = None
    do_baseline = "--baseline" in args
    do_compare = "--compare" in args

    for a in args:
        if a in EVAL_SUITES and not a.startswith("-"):
            only = a

    # Determine which suites to run
    if only:
        suites_to_run = [only]
    else:
        suites_to_run = list(EVAL_SUITES.keys())

    # Run
    results = []
    for key in suites_to_run:
        results.append(_run_suite(key))

    # Always show dashboard
    _print_dashboard(results)

    # Save baseline if requested
    if do_baseline:
        path = _save_baseline(results)
        print(_dim(f"  Baseline saved: {path}"))

    # Compare if requested
    if do_compare:
        _print_comparison(results)

    # Exit code
    total_failed = sum(r["failed"] for r in results)
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
