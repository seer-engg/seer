#!/usr/bin/env python3
"""Download BFCL v3 dataset files from HuggingFace.

Usage:
    python scripts/download_bfcl.py

Downloads BFCL v3 JSON test + possible_answer files into tests/eval/bfcl/data/.
Files are saved with BFCL_v3_ prefix stripped for convenience.
"""
from __future__ import annotations

import urllib.request
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "eval" / "bfcl" / "data"

BASE_URL = (
    "https://huggingface.co/datasets/gorilla-llm/"
    "Berkeley-Function-Calling-Leaderboard/resolve/main"
)

# (remote filename, local filename)
TEST_FILES = [
    ("BFCL_v3_simple.json", "simple.json"),
    ("BFCL_v3_multiple.json", "multiple.json"),
    ("BFCL_v3_parallel.json", "parallel.json"),
    ("BFCL_v3_parallel_multiple.json", "parallel_multiple.json"),
    ("BFCL_v3_live_simple.json", "live_simple.json"),
    ("BFCL_v3_live_multiple.json", "live_multiple.json"),
    ("BFCL_v3_live_parallel.json", "live_parallel.json"),
    ("BFCL_v3_live_parallel_multiple.json", "live_parallel_multiple.json"),
    ("BFCL_v3_multi_turn_base.json", "multi_turn_base.json"),
    ("BFCL_v3_multi_turn_composite.json", "multi_turn_composite.json"),
    ("BFCL_v3_multi_turn_long_context.json", "multi_turn_long_context.json"),
    ("BFCL_v3_multi_turn_miss_param.json", "multi_turn_miss_param.json"),
    ("BFCL_v3_irrelevance.json", "irrelevance.json"),
    ("BFCL_v3_java.json", "java.json"),
    ("BFCL_v3_javascript.json", "javascript.json"),
    ("BFCL_v3_sql.json", "sql.json"),
    ("BFCL_v3_rest.json", "rest.json"),
]

ANSWER_FILES = [
    ("possible_answer/BFCL_v3_simple.json", "answers_simple.json"),
    ("possible_answer/BFCL_v3_multiple.json", "answers_multiple.json"),
    ("possible_answer/BFCL_v3_parallel.json", "answers_parallel.json"),
    ("possible_answer/BFCL_v3_parallel_multiple.json", "answers_parallel_multiple.json"),
    ("possible_answer/BFCL_v3_live_multiple.json", "answers_live_multiple.json"),
    ("possible_answer/BFCL_v3_live_parallel.json", "answers_live_parallel.json"),
    ("possible_answer/BFCL_v3_live_parallel_multiple.json", "answers_live_parallel_multiple.json"),
    ("possible_answer/BFCL_v3_java.json", "answers_java.json"),
    ("possible_answer/BFCL_v3_javascript.json", "answers_javascript.json"),
    ("possible_answer/BFCL_v3_sql.json", "answers_sql.json"),
    ("possible_answer/BFCL_v3_rest.json", "answers_rest.json"),
]


def _download(remote_name: str, local_name: str) -> tuple[int, int, int]:
    """Download one file. Returns (downloaded, skipped, failed)."""
    dest = DATA_DIR / local_name
    if dest.exists():
        print(f"  SKIP {local_name} (exists)")
        return 0, 1, 0

    url = f"{BASE_URL}/{remote_name}"
    print(f"  GET  {local_name}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size = dest.stat().st_size
        print(f"{size:,} bytes")
        return 1, 0, 0
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return 0, 0, 1


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = failed = 0

    print("Test data:")
    for remote, local in TEST_FILES:
        d, s, f = _download(remote, local)
        downloaded += d; skipped += s; failed += f

    print("\nGround truth answers:")
    for remote, local in ANSWER_FILES:
        d, s, f = _download(remote, local)
        downloaded += d; skipped += s; failed += f

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")

    existing = sorted(DATA_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in existing)
    print(f"\n{len(existing)} files in {DATA_DIR} ({total_size:,} bytes total)")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
