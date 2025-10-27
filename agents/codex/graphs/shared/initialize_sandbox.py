from __future__ import annotations

import os


def ensure_sandbox_ready(repo_path: str) -> None:
    if not repo_path:
        raise ValueError("repo_path is required for local runs")
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Repository path not found: {repo_path}")
