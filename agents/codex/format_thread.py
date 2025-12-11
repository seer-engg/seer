# Uses Langfuse API for fetching thread timelines
from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Any
import json
import os
import asyncio
import random
import httpx
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


async def _fetch_langfuse_traces(
    session_id: str | None = None,
    trace_id: str | None = None,
    project_name: str | None = None,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """Fetch traces from Langfuse API"""
    if not config.langfuse_secret_key:
        raise ValueError("LANGFUSE_SECRET_KEY not configured")
    
    headers = {
        "Authorization": f"Bearer {config.langfuse_secret_key}",
        "Content-Type": "application/json"
    }
    
    filter_dict: Dict[str, Any] = {}
    if session_id:
        filter_dict["sessionId"] = {"$eq": session_id}
    if trace_id:
        filter_dict["id"] = {"$eq": trace_id}
    if project_name:
        # Filter by metadata.project_name for metadata-based filtering
        filter_dict["metadata"] = {"project_name": {"$eq": project_name}}
    
    params = {"limit": limit, "page": 1}
    if filter_dict:
        import json as json_lib
        params["filter"] = json_lib.dumps(filter_dict)
    
    delay_seconds = 0.5
    max_attempts = 6
    
    async with httpx.AsyncClient() as client:
        for attempt in range(max_attempts):
            try:
                response = await client.get(
                    f"{config.langfuse_base_url}/api/public/traces",
                    headers=headers,
                    params=params,
                    timeout=30.0
                )
                
                if response.status_code == 429:  # Rate limited
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay_seconds + random.random() * 0.25)
                        delay_seconds *= 2
                        continue
                
                response.raise_for_status()
                data = response.json()
                return data.get("traces", [])
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_attempts - 1:
                    await asyncio.sleep(delay_seconds + random.random() * 0.25)
                    delay_seconds *= 2
                    continue
                raise
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(delay_seconds + random.random() * 0.25)
                delay_seconds *= 2


async def fetch_thread_runs(thread_id: str, project_name: str | None = None):
    """
    Returns a dict: {run_id: run} and a parent->children index for the whole thread.
    """
    if project_name is None:
        project_name = config.langfuse_project_name
        if not project_name:
            raise ValueError(
                "Langfuse project not set. Pass project_name or set LANGFUSE_PROJECT_NAME."
            )

    # Fetch traces by session_id (thread_id maps to session_id in Langfuse)
    root_traces = await _fetch_langfuse_traces(
        session_id=thread_id,
        project_name=project_name,
        limit=1000
    )

    # Fetch observations for each trace to build the tree
    all_runs: Dict[str, Dict[str, Any]] = {}
    children_index: Dict[str, List[str]] = {}
    root_ids: List[str] = []

    if not config.langfuse_secret_key:
        raise ValueError("LANGFUSE_SECRET_KEY not configured")
    
    headers = {
        "Authorization": f"Bearer {config.langfuse_secret_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        for trace in root_traces:
            trace_id = trace.get("id", "")
            if not trace_id:
                continue
            
            root_ids.append(trace_id)
            
            # Fetch observations for this trace
            try:
                obs_response = await client.get(
                    f"{config.langfuse_base_url}/api/public/traces/{trace_id}/observations",
                    headers=headers,
                    params={"limit": 1000},
                    timeout=30.0
                )
                
                if obs_response.status_code == 200:
                    obs_data = obs_response.json()
                    observations = obs_data.get("observations", [])
                    
                    # Add trace as root run
                    all_runs[trace_id] = {
                        "id": trace_id,
                        "run_type": "trace",
                        "name": trace.get("name", "Unnamed"),
                        "start_time": datetime.fromisoformat(trace["startTime"].replace('Z', '+00:00')) if trace.get("startTime") else None,
                        "end_time": datetime.fromisoformat(trace["endTime"].replace('Z', '+00:00')) if trace.get("endTime") else None,
                        "inputs": trace.get("input"),
                        "outputs": trace.get("output"),
                        "error": trace.get("error"),
                        "parent_run_id": None,
                    }
                    children_index[trace_id] = []
                    
                    # Add observations as child runs
                    for obs in observations:
                        obs_id = obs.get("id", "")
                        if not obs_id:
                            continue
                        
                        parent_id = obs.get("parentObservationId") or trace_id
                        
                        all_runs[obs_id] = {
                            "id": obs_id,
                            "run_type": obs.get("type", "span").lower(),
                            "name": obs.get("name", "Unnamed"),
                            "start_time": datetime.fromisoformat(obs["startTime"].replace('Z', '+00:00')) if obs.get("startTime") else None,
                            "end_time": datetime.fromisoformat(obs["endTime"].replace('Z', '+00:00')) if obs.get("endTime") else None,
                            "inputs": obs.get("input"),
                            "outputs": obs.get("output"),
                            "error": obs.get("error"),
                            "parent_run_id": parent_id,
                        }
                        
                        if parent_id not in children_index:
                            children_index[parent_id] = []
                        children_index[parent_id].append(obs_id)
                        children_index.setdefault(obs_id, [])
            except Exception as e:
                # If we can't fetch observations, at least include the trace
                all_runs[trace_id] = {
                    "id": trace_id,
                    "run_type": "trace",
                    "name": trace.get("name", "Unnamed"),
                    "start_time": datetime.fromisoformat(trace["startTime"].replace('Z', '+00:00')) if trace.get("startTime") else None,
                    "end_time": datetime.fromisoformat(trace["endTime"].replace('Z', '+00:00')) if trace.get("endTime") else None,
                    "inputs": trace.get("input"),
                    "outputs": trace.get("output"),
                    "error": trace.get("error"),
                    "parent_run_id": None,
                }
                children_index[trace_id] = []

    return all_runs, children_index, root_ids

async def fetch_thread_timeline_as_string(thread_id: str, project_name: str | None = None):
    runs, children, root_ids = await fetch_thread_runs(thread_id, project_name)

    lines: List[str] = []
    lines.append("=== Nested view by root ===")

    def walk(run_id: str, depth=0):
        r = runs[run_id]
        indent = "  " * depth
        run_name = r.get("name") or r.get("run_type", "unknown")
        run_type = r.get("run_type", "unknown")
        lines.append(f"{indent}-{run_name} (type={run_type})")
        if r.get("inputs"):
            lines.append(f"{indent}    ↳ in : {_short(r['inputs'])}")
        if r.get("outputs"):
            lines.append(f"{indent}    ↳ out: {_short(r['outputs'])}")
        for kid in sorted(children[run_id], key=lambda k: (runs[k].get("start_time") or runs[k].get("end_time") or datetime.min)):
            walk(kid, depth + 1)

    if not root_ids:
        lines.append(f"(no runs found for thread: {thread_id})")
        return "\n".join(lines)

    for root in sorted(root_ids, key=lambda k: (runs[k].get("start_time") or runs[k].get("end_time") or datetime.min)):
        walk(root, 0)

    return "\n".join(lines)

