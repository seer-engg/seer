"""Workflow run history, checkpoints, and result retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from seer.api.agents.checkpointer import get_checkpointer
from seer.api.workflows import models as api_models

from seer.services.workflows.execution import _build_run_config, _compile_workflow
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
    ForEachNode,
    IfNode,
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
    enriched["output_key"] = node.out
    if node.expect_output:
        enriched["expect_output"] = node.expect_output.model_dump() if hasattr(node.expect_output, "model_dump") else node.expect_output


def _enrich_with_llm_node(enriched: Dict[str, Any], node: LLMNode) -> None:
    """Enrich trace data with LLMNode metadata."""
    enriched["model"] = node.model
    enriched["output_key"] = node.out
    if node.prompt:
        enriched["prompt_template"] = node.prompt
    if node.temperature is not None:
        enriched["temperature"] = node.temperature
    if node.output:
        enriched["output_schema"] = node.output.model_dump() if hasattr(node.output, "model_dump") else node.output


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


async def _extract_node_traces_from_graph(
    user: User,
    run: WorkflowRun,
    checkpointer: Any,
    config: Dict[str, Any],
    workflow_spec: Optional[WorkflowSpec],
) -> List[Dict[str, Any]]:
    """Extract node traces from compiled graph state."""
    nodes = []
    trace_keys_found = set()

    logger.info(
        "Compiling workflow to access full state for thread_id '%s'",
        config.get('configurable', {}).get('thread_id'),
        extra={"run_id": run.run_id}
    )

    compiled = await _compile_workflow(user, run.spec, checkpointer=checkpointer)

    graph_state = await compiled.workflow.graph.aget_state(config)

    if not graph_state or not graph_state.values:
        logger.warning(
            "No state values found from graph.aget_state() for run '%s'",
            run.run_id,
            extra={"run_id": run.run_id}
        )
        return nodes

    state_values = graph_state.values
    logger.info(
        "Retrieved full state from graph with %s keys",
        len(state_values),
        extra={"run_id": run.run_id, "state_keys": list(state_values.keys())[:20]}
    )

    # Extract trace keys from full state
    for key, value in state_values.items():
        if not key.startswith("_trace_"):
            continue

        node_id = key.replace("_trace_", "")
        if node_id in trace_keys_found:
            continue

        trace_keys_found.add(node_id)
        if not isinstance(value, dict):
            continue

        node_trace = {
            "node_id": node_id,
            "node_type": value.get("node_type", "unknown"),
            "inputs": value.get("inputs", {}),
            "output": value.get("output"),
            "timestamp": value.get("timestamp"),
            "output_key": value.get("output_key"),
        }

        try:
            enriched_node = _enrich_node_with_spec(node_trace, node_id, workflow_spec)
            nodes.append(enriched_node)
            logger.info(
                "Found trace data for node '%s' in graph state",
                node_id,
                extra={"run_id": run.run_id, "node_id": node_id}
            )
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


async def _extract_node_traces_from_checkpoints(
    checkpointer: Any,
    config: Dict[str, Any],
    run: WorkflowRun,
    workflow_spec: Optional[WorkflowSpec],
) -> List[Dict[str, Any]]:
    """Fallback: Extract node traces from checkpoint channel_values."""
    nodes = []
    trace_keys_found = set()

    async for checkpoint_tuple in checkpointer.alist(config):
        checkpoint = checkpoint_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {})

        for key, value in channel_values.items():
            if not key.startswith("_trace_"):
                continue

            node_id = key.replace("_trace_", "")
            if node_id in trace_keys_found:
                continue

            trace_keys_found.add(node_id)
            if not isinstance(value, dict):
                continue

            node_trace = {
                "node_id": node_id,
                "node_type": value.get("node_type", "unknown"),
                "inputs": value.get("inputs", {}),
                "output": value.get("output"),
                "timestamp": value.get("timestamp"),
                "output_key": value.get("output_key"),
            }

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


async def _validate_history_prerequisites() -> None:
    """Validate that history retrieval prerequisites are met."""
    if not shared_config.DATABASE_URL:
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


async def _extract_node_traces_with_fallback(
    user: User,
    run: WorkflowRun,
    checkpointer: Any,
    config: Dict[str, Any],
    workflow_spec: Optional[WorkflowSpec],
) -> List[Dict[str, Any]]:
    """Extract node traces from graph with fallback to checkpoints."""
    try:
        return await _extract_node_traces_from_graph(
            user, run, checkpointer, config, workflow_spec
        )
    except Exception as graph_exc:
        logger.warning(
            "Error accessing graph state, falling back to checkpoint channel_values: %s",
            graph_exc,
            exc_info=True,
            extra={"run_id": run.run_id}
        )
        try:
            return await _extract_node_traces_from_checkpoints(
                checkpointer, config, run, workflow_spec
            )
        except Exception as list_exc:
            logger.error(
                "Error in fallback checkpoint iteration: %s",
                list_exc,
                exc_info=True,
                extra={"run_id": run.run_id}
            )
            return []


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

        nodes = await _extract_node_traces_with_fallback(
            user, run, checkpointer, config, workflow_spec
        )

        logger.info(
            "Found %s node trace(s) for run '%s'",
            len(nodes),
            run.run_id,
            extra={"run_id": run.run_id, "node_count": len(nodes), "node_ids": [n.get("node_id") for n in nodes]}
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
