# pylint: disable=broad-exception-caught,import-outside-toplevel,consider-using-f-string
# Reason: History service needs broad exception handling for checkpoint operations; lazy imports; legacy % formatting
"""Workflow run history, checkpoints, and result retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from seer.api.agents.checkpointer import get_checkpointer
from seer.api.workflows import models as api_models

from seer.services.workflows.execution import _build_run_config
from seer.api.core.errors import RUN_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.config import config as shared_config
from seer.database import (
    User,
    WorkflowRun,
    WorkflowRunStatus,
    parse_run_public_id,
)
from seer.logger import get_logger
from seer.core.schema.models import (
    LLMNode,
    Node,
    ToolNode,
    WorkflowSpec,
)

logger = get_logger(__name__)


def _snapshot_to_dict(snapshot: Any) -> Dict[str, Any]:
    serializable: Dict[str, Any] = {}
    for key in (
        "checkpoint_id",
        "parent_checkpoint_id",
        "values",
        "next",
        "tasks",
        "metadata",
        "created_at",
        "config",
        "parent_config",
    ):
        if hasattr(snapshot, key):
            value = getattr(snapshot, key)
            if value is not None:
                serializable[key] = value
    return serializable


def _find_node_in_spec(nodes: List[Node], target_id: str) -> Optional[Node]:
    """Find a node by ID in the workflow spec."""
    for node in nodes:
        if node.id == target_id:
            return node
    return None


def _enrich_with_tool_node(enriched: Dict[str, Any], node: ToolNode) -> None:
    """Enrich trace data with ToolNode metadata."""
    enriched["tool_name"] = node.tool
    if node.expect_outputs:
        enriched["expect_outputs"] = node.expect_outputs.model_dump() if hasattr(node.expect_outputs, "model_dump") else node.expect_outputs


def _enrich_with_llm_node(enriched: Dict[str, Any], node: LLMNode) -> None:
    """Enrich trace data with LLMNode metadata."""
    if "model" in node.inputs:
        enriched["model"] = node.inputs["model"]
    if "prompt" in node.inputs:
        enriched["prompt_template"] = node.inputs["prompt"]
    if "temperature" in node.inputs:
        enriched["temperature"] = node.inputs["temperature"]
    if node.outputs:
        enriched["output_schema"] = node.outputs.model_dump() if hasattr(node.outputs, "model_dump") else node.outputs


def _enrich_node_with_spec(
    node_trace: Dict[str, Any],
    node_id: str,
    workflow_spec: Optional[WorkflowSpec],
) -> Dict[str, Any]:
    """Enrich node trace with workflow spec metadata."""
    enriched = node_trace.copy()

    if not workflow_spec:
        return enriched

    node = _find_node_in_spec(workflow_spec.nodes, node_id)
    if not node:
        return enriched

    if isinstance(node, ToolNode):
        _enrich_with_tool_node(enriched, node)
    elif isinstance(node, LLMNode):
        _enrich_with_llm_node(enriched, node)

    return enriched


def _build_node_label(node: Node) -> str:
    """Build display label for a node."""
    node_id = node.id
    if isinstance(node, ToolNode):
        return f"{node_id} ({node.tool})"
    if isinstance(node, LLMNode):
        return f"{node_id} (LLM)"
    return node_id


def _collect_graph_nodes(node: Node, nodes: List[Dict[str, Any]]) -> None:
    """Collect node information for graph visualization."""
    node_id = node.id
    node_type = node.type if hasattr(node, "type") else "unknown"

    nodes.append({
        "id": node_id,
        "type": node_type,
        "label": _build_node_label(node),
    })


def _build_execution_graph(workflow_spec: Optional[WorkflowSpec]) -> Dict[str, Any]:
    """Build execution graph structure from workflow spec."""
    if not workflow_spec:
        return {"nodes": [], "edges": []}

    nodes = []
    for node in workflow_spec.nodes:
        _collect_graph_nodes(node, nodes)

    edges = [
        {
            "source": edge.source,
            "target": edge.target,
        }
        for edge in workflow_spec.edges
    ]

    return {"nodes": nodes, "edges": edges}


def _serialize_datetime(dt: Optional[Any]) -> Optional[str]:
    """Safely serialize a datetime to ISO format string."""
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except (AttributeError, ValueError):
        return None


async def _fetch_checkpoint_state(
    checkpointer: Any,
    config: Dict[str, Any],
    run: WorkflowRun,
) -> Any:
    """Fetch checkpoint state, retrying with explicit namespace if needed."""
    # Try to get checkpoint
    state_tuple = await checkpointer.aget_tuple(config)
    if state_tuple:
        return state_tuple

    # Try with explicit checkpoint_ns=""
    config_with_ns = dict(config)
    config_with_ns.setdefault("configurable", {})["checkpoint_ns"] = ""
    logger.info(
        "Retrying checkpoint retrieval with explicit checkpoint_ns='' for run '%s'",
        run.run_id,
        extra={"run_id": run.run_id, "config_with_ns": config_with_ns}
    )
    state_tuple = await checkpointer.aget_tuple(config_with_ns)
    if state_tuple:
        logger.info(
            "Checkpoint found with checkpoint_ns='' for run '%s'",
            run.run_id,
            extra={"run_id": run.run_id}
        )
    return state_tuple


def _extract_node_id_from_trace_key(key: str) -> tuple[str, str]:
    """
    Extract node_id from trace key.

    Returns:
        Tuple of (node_id, unique_trace_id)
    """
    # Handle both _trace_{node_id} and _trace_{node_id}_iter_{N}
    if "_iter_" in key:
        # Loop iteration trace: _trace_{node_id}_iter_{N}
        node_id = key.replace("_trace_", "").split("_iter_")[0]
        # For loop iterations, use full key as unique identifier
        unique_trace_id = key
    else:
        # Regular trace: _trace_{node_id}
        node_id = key.replace("_trace_", "")
        unique_trace_id = node_id
    return node_id, unique_trace_id


def _build_node_trace_from_value(value: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Build node trace dict from checkpoint value."""
    node_trace = {
        "node_id": node_id,
        "node_type": value.get("node_type", "unknown"),
        "inputs": value.get("inputs", {}),
        "output": value.get("output"),
        "timestamp": value.get("timestamp"),
        "output_key": value.get("output_key"),
    }

    # Include status if present (succeeded/failed from error trace implementation)
    if "status" in value:
        node_trace["status"] = value["status"]

    # Include error info if present (for failed nodes)
    if "error" in value:
        node_trace["error"] = value["error"]

    # Include LLM usage data if present
    if "usage" in value:
        node_trace["usage"] = value["usage"]

    return node_trace


async def _extract_node_traces_from_checkpoints(
    checkpointer: Any,
    config: Dict[str, Any],
    run: WorkflowRun,
    workflow_spec: Optional[WorkflowSpec],
) -> List[Dict[str, Any]]:
    """Extract node traces from checkpoint channel_values."""
    nodes = []
    trace_keys_found = set()

    async for checkpoint_tuple in checkpointer.alist(config):
        checkpoint = checkpoint_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {})

        # CRITICAL FIX: LangGraph stores state in __root__ channel for Dict[str, Any] schema
        # Traces are at: checkpoint["channel_values"]["__root__"]["_trace_*"]
        # Not at: checkpoint["channel_values"]["_trace_*"]
        state_dict = channel_values.get("__root__", channel_values)

        for key, value in state_dict.items():
            if not key.startswith("_trace_"):
                continue

            node_id, unique_trace_id = _extract_node_id_from_trace_key(key)

            # Skip if we already processed this exact trace
            if unique_trace_id in trace_keys_found:
                continue

            trace_keys_found.add(unique_trace_id)
            if not isinstance(value, dict):
                continue

            node_trace = _build_node_trace_from_value(value, node_id)

            try:
                enriched_node = _enrich_node_with_spec(node_trace, node_id, workflow_spec)
                nodes.append(enriched_node)
            except Exception as enrich_exc:
                logger.warning(
                    "Failed to enrich node '%s' with spec metadata: %s",
                    node_id,
                    enrich_exc,
                    exc_info=True,
                    extra={"run_id": run.run_id, "node_id": node_id}
                )
                nodes.append(node_trace)

    return nodes


def _get_error_traces_from_database(run: WorkflowRun) -> List[Dict[str, Any]]:
    """
    Extract error traces from the database (node_traces column).

    Failed nodes don't get checkpointed by LangGraph since the exception is raised
    before the node can return. Instead, we persist error traces directly to the
    database when ExecutionError is raised.

    Returns:
        List of error trace dicts from database, empty if none available.
    """
    if not run.node_traces:
        return []

    db_traces = []
    # node_traces can be a dict like {"_trace_node_id": {trace_data}} or just trace_data directly
    if isinstance(run.node_traces, dict):
        # Check if it's a single trace (has 'node_id' key) or a collection of traces
        if 'node_id' in run.node_traces:
            # Single trace dict
            db_traces.append(run.node_traces)
        else:
            # Collection of traces keyed by trace_key
            for trace in run.node_traces.values():
                if isinstance(trace, dict) and trace.get('status') == 'failed':
                    db_traces.append(trace)

    return db_traces


# =============================================================================
# BUG FIX: History Trace Merge - Show Failed Traces (2024-02 RCA)
# =============================================================================
# PROBLEM: When debugging workflow failures, the history API only showed the
#   successful trace, hiding the failed trace that caused the actual error.
#   User saw "google_sheets_write succeeded" but logs showed it failed.
#
# ROOT CAUSE: The original merge logic preferred checkpoint data over database
#   traces. If a node succeeded in ANY execution, the success trace would hide
#   the failure trace from previous attempts:
#
# ORIGINAL (BUGGY) CODE:
#   checkpoint_node_ids = {t.get('node_id') for t in checkpoint_traces}
#   for trace in db_traces:
#       if node_id not in checkpoint_node_ids:  # <-- Skipped if success existed!
#           merged.append(trace)
#
# WHY NOT CAUGHT: Tests focused on the "happy path" - either all succeed or all
#   fail. No tests for partial success scenarios (succeed then fail, or vice versa).
#
# FIX: Always include failed traces, regardless of whether a success exists.
#   This is essential for debugging:
#   - Loop iterations where node fails on some iterations but succeeds on others
#   - Retry scenarios where earlier attempts failed but later succeeded
#   - Race conditions in parallel execution
# =============================================================================
def _merge_checkpoint_and_database_traces(
    checkpoint_traces: List[Dict[str, Any]],
    db_traces: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge traces from checkpoints (successful nodes) with database traces (failed nodes).

    Checkpoint traces are authoritative for successful executions.
    Database traces fill in the gaps for failed nodes that weren't checkpointed.

    IMPORTANT: Failed traces are ALWAYS included, even if a successful trace exists
    for the same node. This is critical for debugging loop iterations where the same
    node may succeed on some iterations but fail on others.

    Returns:
        Merged list of all node traces, including both successful and failed attempts.
    """
    merged = list(checkpoint_traces)

    # -------------------------------------------------------------------------
    # CRITICAL: Always include failed traces for debugging visibility
    # -------------------------------------------------------------------------
    # Without this, users see confusing output like:
    #   "log_sent_status succeeded" (from checkpoint)
    # But the actual run failed because a DIFFERENT iteration's log_sent_status
    # failed with RemoteProtocolError (stored in database only).
    # -------------------------------------------------------------------------
    for trace in db_traces:
        if trace.get('status') == 'failed':
            # Always include failed traces for debugging
            merged.append(trace)
        else:
            # For non-failed db traces, only add if not already in checkpoint traces
            node_id = trace.get('node_id')
            checkpoint_node_ids = {t.get('node_id') for t in checkpoint_traces if t.get('node_id')}
            if node_id and node_id not in checkpoint_node_ids:
                merged.append(trace)

    return merged


async def _validate_history_prerequisites() -> None:
    """Validate that history retrieval prerequisites are met."""
    if not shared_config.database_url:
        raise_problem(
            type_uri=RUN_PROBLEM,
            title="History unavailable",
            detail="LangGraph checkpointer is not configured",
            status=503,
        )


async def _get_checkpointer_or_fail() -> Any:
    """Get checkpointer instance or raise error."""
    checkpointer = await get_checkpointer()
    if checkpointer is None:
        raise_problem(
            type_uri=RUN_PROBLEM,
            title="History unavailable",
            detail="LangGraph checkpointer failed to initialize",
            status=503,
        )
    return checkpointer


def _parse_workflow_spec(run: WorkflowRun) -> Optional[WorkflowSpec]:
    """Parse workflow spec from run, returning None if parsing fails."""
    try:
        return WorkflowSpec.model_validate(run.spec)
    except Exception:
        return None


def _build_history_response(
    run: WorkflowRun,
    nodes: List[Dict[str, Any]],
    workflow_spec: Optional[WorkflowSpec],
) -> List[Dict[str, Any]]:
    """Build history response dictionary."""
    execution_graph = _build_execution_graph(workflow_spec)
    workflow_id = run.workflow.workflow_id if run.workflow else None

    return [{
        "run_id": run.run_id,
        "workflow_id": workflow_id,
        "status": run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
        "created_at": _serialize_datetime(run.created_at),
        "started_at": _serialize_datetime(run.started_at),
        "finished_at": _serialize_datetime(run.finished_at),
        "nodes": nodes,
        "execution_graph": execution_graph,
    }]


async def _get_run(user: User, run_id: str) -> WorkflowRun:
    """
    Get a workflow run by ID, prefetching the workflow relationship.

    Prefetches the workflow ForeignKey to avoid QuerySet issues when accessing
    run.workflow.workflow_id later.
    """
    try:
        pk = parse_run_public_id(run_id)
    except ValueError:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid run id",
            detail="Run id is invalid",
            status=400,
        )
    # Use filter().prefetch_related().first() instead of .get() to prefetch workflow
    run = await WorkflowRun.filter(id=pk, user=user).prefetch_related('workflow').first()
    if not run:
        raise_problem(
            type_uri=RUN_PROBLEM,
            title="Run not found",
            detail=f"Run '{run_id}' not found",
            status=404,
        )
    return run


async def get_run_history(user: User, run_id: str) -> api_models.RunHistoryResponse:
    """
    Get workflow execution history with node-based traces.

    Returns node execution traces extracted from the latest checkpoint,
    enriched with workflow spec metadata.
    """
    await _validate_history_prerequisites()
    run = await _get_run(user, run_id)
    checkpointer = await _get_checkpointer_or_fail()
    workflow_spec = _parse_workflow_spec(run)

    config = _build_run_config(run, run.config)
    logger.info(
        "Retrieving checkpoint for run '%s' with config: %s",
        run.run_id,
        config,
        extra={
            "run_id": run.run_id,
            "config": config,
            "run_config": run.config,
            "run_status": run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
        }
    )

    try:
        state_tuple = await _fetch_checkpoint_state(checkpointer, config, run)
        if not state_tuple:
            raise_problem(
                type_uri=RUN_PROBLEM,
                title="Run history not found",
                detail="No checkpoints found for run '%s'" % run.run_id,
                status=404,
            )

        checkpoint_traces = await _extract_node_traces_from_checkpoints(
            checkpointer, config, run, workflow_spec
        )

        # Get error traces from database (for failed nodes that weren't checkpointed)
        db_traces = _get_error_traces_from_database(run)

        # Merge checkpoint traces (successful nodes) with database traces (failed nodes)
        nodes = _merge_checkpoint_and_database_traces(checkpoint_traces, db_traces)

        logger.info(
            "Found %s node trace(s) for run '%s' (checkpoint: %s, database: %s)",
            len(nodes),
            run.run_id,
            len(checkpoint_traces),
            len(db_traces),
            extra={
                "run_id": run.run_id,
                "node_count": len(nodes),
                "checkpoint_trace_count": len(checkpoint_traces),
                "db_trace_count": len(db_traces),
                "node_ids": [n.get("node_id") for n in nodes],
            }
        )

        history = _build_history_response(run, nodes, workflow_spec)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error loading run history for run '%s': %s",
            run.run_id, exc,
            exc_info=True,
            extra={"run_id": run.run_id, "user_id": user.id}
        )
        raise_problem(
            type_uri=RUN_PROBLEM,
            title="Failed to load run history",
            detail=f"An error occurred while loading history for run '{run.run_id}': {str(exc)}",
            status=500,
        )

    return api_models.RunHistoryResponse(run_id=run.run_id, history=history)


async def get_run_status(user: User, run_id: str) -> api_models.RunResponse:
    run = await _get_run(user, run_id)
    from seer.api.workflows.services.execution import _serialize_run
    return _serialize_run(run)
