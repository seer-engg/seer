# pip install langsmith
from __future__ import annotations
from langsmith import AsyncClient
from datetime import datetime
from typing import Dict, List
import json
import os
import asyncio
import random
from langsmith.schemas import Run
from shared.config import config

MAX_IO_CHARS = 10000  # keep console readable

KEYS_TO_REMOVE = {"id", "model_name", "refusal", "logprobs", "model_provider", "service_tier",
"system_fingerprint", "token_usage" , "usage_metadata",
"additional_kwargs"
}

def _remove_keys_recursively(value, keys_to_remove={"id", "model_name"}):
    """
    Return a new structure with specified keys removed from all nested dicts.
    Does not mutate the original input.
    """
    if isinstance(value, dict):
        return {
            key: _remove_keys_recursively(nested_value, keys_to_remove)
            for key, nested_value in value.items()
            if key not in keys_to_remove
        }
    if isinstance(value, list):
        return [_remove_keys_recursively(item, keys_to_remove) for item in value]
    if isinstance(value, tuple):
        return tuple(_remove_keys_recursively(item, keys_to_remove) for item in value)
    return value

def _short(obj:dict):
    try:
        sanitized = _remove_keys_recursively(obj, KEYS_TO_REMOVE)
        s = json.dumps(sanitized, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return (s[:MAX_IO_CHARS] + "…") if s and len(s) > MAX_IO_CHARS else s


async def _collect_runs(client: AsyncClient, **kwargs) -> List[Run]:
    delay_seconds = 0.5
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            return [r async for r in client.list_runs(**kwargs)]
        except Exception as e:
            msg = str(e)
            is_rate_limited = "Rate limit" in msg or "429" in msg
            if not is_rate_limited or attempt == max_attempts - 1:
                raise
            await asyncio.sleep(delay_seconds + random.random() * 0.25)
            delay_seconds *= 2


async def fetch_thread_runs(thread_id: str, project_name: str | None = None):
    """
    Returns a dict: {run_id: run} and a parent->children index for the whole thread.
    """
    client = AsyncClient()
    # 1) Find root runs that belong to this conversational thread

    # Ensure we scope the query to a project to satisfy LangSmith API requirements
    if project_name is None:
        project_name = config.target_agent_langsmith_project
        if not project_name:
            raise ValueError(
                "LangSmith project not set. Pass project_name or set LANGSMITH_PROJECT."
            )

    group_key = thread_id
    filter_string = f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "{group_key}"))'
    roots = await _collect_runs(
        client,
        project_name=project_name,
        filter=filter_string,
        is_root=True,
        limit=1000,
        select=[
            "id",
            "run_type",
            "name",
            "start_time",
            "end_time",
            "inputs",
            "outputs",
            "error",
            "parent_run_id",
        ],
    )

    # 2) Walk the tree to get every descendant
    all_runs: Dict[str, Run] = {}
    children_index: Dict[str, List[str]] = {}

    for root in roots:
        runs_in_trace = await _collect_runs(
            client,
            project_name=project_name,
            trace_id=root.id,
            limit=1000,
            select=[
                "id",
                "run_type",
                "name",
                "start_time",
                "end_time",
                "inputs",
                "outputs",
                "error",
                "parent_run_id",
            ],
        )
        for run in runs_in_trace:
            rid = str(run.id)
            all_runs[rid] = run
            if run.parent_run_id:
                pid = str(run.parent_run_id)
                children_index.setdefault(pid, []).append(rid)
            children_index.setdefault(rid, [])

    return all_runs, children_index, [str(r.id) for r in roots]

async def fetch_thread_timeline_as_string(thread_id: str, project_name: str | None = None):
    runs, children, root_ids = await fetch_thread_runs(thread_id, project_name)

    lines: List[str] = []
    lines.append("=== Nested view by root ===")

    def walk(run_id: str, depth=0):
        r = runs[run_id]
        indent = "  " * depth
        lines.append(f"{indent}-{r.name or r.run_type} (type={r.run_type})")
        if r.inputs:
            lines.append(f"{indent}    ↳ in : {_short(r.inputs)}")
        if r.outputs:
            lines.append(f"{indent}    ↳ out: {_short(r.outputs)}")
        for kid in sorted(children[run_id], key=lambda k: (runs[k].start_time or runs[k].end_time)):
            walk(kid, depth + 1)

    if not root_ids:
        lines.append(f"(no runs found for thread: {thread_id})")
        return "\n".join(lines)

    for root in sorted(root_ids, key=lambda k: (runs[k].start_time or runs[k].end_time)):
        walk(root, 0)

    return "\n".join(lines)

