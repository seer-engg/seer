from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import json
import os
import asyncio
import mlflow
from shared.config import config
from shared.logger import get_logger

logger = get_logger("codex.format_thread")

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


async def _fetch_mlflow_traces(
    session_id: str | None = None,
    trace_id: str | None = None,
    project_name: str | None = None,
    limit: int = 1000
) -> List[Any]:
    """
    Fetch traces from MLflow using the MLflow SDK.
    
    Args:
        session_id: Thread/session ID to filter by (maps to tags.session_id)
        trace_id: Specific trace ID to fetch
        project_name: Project name for filtering (can use experiment name or tags.project_name)
        limit: Maximum number of traces to return
    
    Returns:
        List of MLflow Trace objects
    """
    if not config.mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI not configured")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    # Build filter string for MLflow search
    filter_parts = []
    
    if session_id:
        filter_parts.append(f"tags.session_id = '{session_id}'")
    
    if trace_id:
        # For specific trace_id, use get_trace instead of search
        try:
            trace = mlflow.get_trace(trace_id)
            return [trace] if trace else []
        except Exception as e:
            logger.warning(f"Failed to get trace by ID {trace_id}: {e}")
            # Fall back to search with trace_id filter
            filter_parts.append(f"trace_id = '{trace_id}'")
    
    if project_name:
        # Try filtering by experiment name first, then fall back to tags
        try:
            experiment = mlflow.get_experiment_by_name(project_name)
            if experiment:
                experiment_ids = [experiment.experiment_id]
            else:
                # Fall back to tag filtering
                filter_parts.append(f"tags.project_name = '{project_name}'")
                experiment_ids = None
        except Exception:
            # Fall back to tag filtering
            filter_parts.append(f"tags.project_name = '{project_name}'")
            experiment_ids = None
    else:
        experiment_ids = None
    
    filter_string = " AND ".join(filter_parts) if filter_parts else None
    
    # Search traces using MLflow SDK
    try:
        traces = mlflow.search_traces(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=limit,
            return_type="list"  # Return Trace objects, not DataFrame
        )
        return traces
    except Exception as e:
        logger.error(f"Error searching MLflow traces: {e}", exc_info=True)
        raise ValueError(f"Failed to fetch traces from MLflow: {str(e)}")


async def fetch_thread_runs(thread_id: str, project_name: str | None = None):
    """
    Returns a dict: {run_id: run} and a parent->children index for the whole thread.
    Uses MLflow traces which contain nested spans.
    """
    if not config.mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI not configured. Set MLFLOW_TRACKING_URI environment variable.")

    # Fetch traces by session_id (thread_id maps to tags.session_id in MLflow)
    root_traces = await _fetch_mlflow_traces(
        session_id=thread_id,
        project_name=project_name,
        limit=1000
    )

    # Build runs dict and children index from MLflow traces and spans
    all_runs: Dict[str, Dict[str, Any]] = {}
    children_index: Dict[str, List[str]] = {}
    root_ids: List[str] = []

    for trace in root_traces:
        try:
            trace_info = trace.info
            trace_id = trace_info.trace_id
            
            if not trace_id:
                logger.warning("Trace missing trace_id, skipping")
                continue
            
            root_ids.append(trace_id)
            
            # Map MLflow status to our format
            status_map = {
                "OK": "success",
                "ERROR": "error",
                "IN_PROGRESS": "pending",
            }
            status = status_map.get(trace_info.status, "pending")
            
            # Convert milliseconds to datetime
            start_time = None
            if trace_info.start_time_ms:
                start_time = datetime.fromtimestamp(trace_info.start_time_ms / 1000, tz=timezone.utc)
            
            end_time = None
            if trace_info.end_time_ms:
                end_time = datetime.fromtimestamp(trace_info.end_time_ms / 1000, tz=timezone.utc)
            
            # Extract inputs/outputs from trace info
            inputs = trace_info.inputs if hasattr(trace_info, 'inputs') else None
            outputs = trace_info.outputs if hasattr(trace_info, 'outputs') else None
            error = trace_info.exception if hasattr(trace_info, 'exception') else None
            
            # Get trace name from request_id or trace_name
            trace_name = getattr(trace_info, 'request_id', None) or getattr(trace_info, 'trace_name', None) or "Unnamed"
            
            # Add trace as root run
            all_runs[trace_id] = {
                "id": trace_id,
                "run_type": "trace",
                "name": trace_name,
                "start_time": start_time,
                "end_time": end_time,
                "duration": (trace_info.end_time_ms - trace_info.start_time_ms) / 1000 if (trace_info.end_time_ms and trace_info.start_time_ms) else None,
                "status": status,
                "inputs": inputs,
                "outputs": outputs,
                "error": str(error) if error else None,
                "parent_run_id": None,
            }
            children_index[trace_id] = []
            
            # Process spans (nested runs)
            if hasattr(trace, 'spans') and trace.spans:
                # Build a map of span_id -> span for quick lookup
                span_map: Dict[str, Any] = {}
                for span in trace.spans:
                    span_map[span.span_id] = span
                
                # Process spans and build parent-child relationships
                for span in trace.spans:
                    span_id = span.span_id
                    if not span_id:
                        continue
                    
                    # Determine parent: if parent_span_id exists, use it; otherwise parent is trace
                    parent_span_id = getattr(span, 'parent_span_id', None)
                    parent_id = parent_span_id if parent_span_id else trace_id
                    
                    # Map span status
                    span_status = status_map.get(getattr(span, 'status', 'OK'), "pending")
                    
                    # Convert span timestamps
                    span_start_time = None
                    if hasattr(span, 'start_time_ms') and span.start_time_ms:
                        span_start_time = datetime.fromtimestamp(span.start_time_ms / 1000, tz=timezone.utc)
                    
                    span_end_time = None
                    if hasattr(span, 'end_time_ms') and span.end_time_ms:
                        span_end_time = datetime.fromtimestamp(span.end_time_ms / 1000, tz=timezone.utc)
                    
                    # Get span type (e.g., "LLM", "TOOL", "CHAIN", etc.)
                    span_type = getattr(span, 'span_type', 'span').lower()
                    
                    # Extract inputs/outputs from span
                    span_inputs = getattr(span, 'inputs', None)
                    span_outputs = getattr(span, 'outputs', None)
                    span_error = getattr(span, 'exception', None)
                    
                    span_name = getattr(span, 'name', 'Unnamed')
                    
                    # Calculate duration
                    span_duration = None
                    if hasattr(span, 'start_time_ms') and hasattr(span, 'end_time_ms'):
                        if span.start_time_ms and span.end_time_ms:
                            span_duration = (span.end_time_ms - span.start_time_ms) / 1000
                    
                    # Add span as child run
                    all_runs[span_id] = {
                        "id": span_id,
                        "run_type": span_type,
                        "name": span_name,
                        "start_time": span_start_time,
                        "end_time": span_end_time,
                        "duration": span_duration,
                        "status": span_status,
                        "inputs": span_inputs,
                        "outputs": span_outputs,
                        "error": str(span_error) if span_error else None,
                        "parent_run_id": parent_id,
                    }
                    
                    # Update children index
                    if parent_id not in children_index:
                        children_index[parent_id] = []
                    children_index[parent_id].append(span_id)
                    children_index.setdefault(span_id, [])
        except Exception as e:
            logger.error(f"Error processing MLflow trace: {e}", exc_info=True)
            # Continue with next trace even if one fails
            continue

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

