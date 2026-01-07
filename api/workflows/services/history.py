"""Workflow run history, checkpoints, and result retrieval."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.workflows import models as api_models
from api.workflows.services._shared import (
    RUN_PROBLEM,
    VALIDATION_PROBLEM,
    _build_run_config,
    _raise_problem,
)
from api.agents.checkpointer import get_checkpointer
from shared.database.base import async_session_maker
from shared.database.models import (
    User,
    WorkflowRun,
    WorkflowRunStatus,
    make_workflow_public_id,
    parse_run_public_id,
)
from workflow_compiler.runtime.global_compiler import WorkflowCompilerSingleton
from workflow_compiler.schema.models import (
    ForEachNode,
    IfNode,
    LLMNode,
    Node,
    ToolNode,
    WorkflowSpec,
)

compiler = WorkflowCompilerSingleton.instance()
logger = logging.getLogger(__name__)


def _snapshot_to_dict(snapshot: Any) -> Dict[str, Any]:
    """Convert checkpoint snapshot to dictionary."""
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


def _enrich_node_with_spec(
    node_trace: Dict[str, Any],
    node_id: str,
    workflow_spec: Optional[WorkflowSpec],
) -> Dict[str, Any]:
    """
    Enrich node trace with workflow spec metadata.

    Args:
        node_trace: Node execution trace data
        node_id: Node identifier
        workflow_spec: Workflow specification

    Returns:
        Enriched node trace with tool/LLM metadata
    """
    enriched = node_trace.copy()

    if workflow_spec:
        # Find node in spec
        def find_node(nodes: List[Node], target_id: str) -> Optional[Node]:
            for node in nodes:
                if node.id == target_id:
                    return node
                # Handle nested nodes in IfNode and ForEachNode
                if isinstance(node, IfNode):
                    found = find_node(node.then, target_id)
                    if found:
                        return found
                    found = find_node(node.else_, target_id)
                    if found:
                        return found
                elif isinstance(node, ForEachNode):
                    found = find_node(node.body, target_id)
                    if found:
                        return found
            return None

        node = find_node(workflow_spec.nodes, node_id)
        if node:
            if isinstance(node, ToolNode):
                enriched["tool_name"] = node.tool
                enriched["output_key"] = node.out
                if node.expect_output:
                    enriched["expect_output"] = node.expect_output.model_dump() if hasattr(node.expect_output, "model_dump") else node.expect_output
            elif isinstance(node, LLMNode):
                enriched["model"] = node.model
                enriched["output_key"] = node.out
                if node.prompt:
                    enriched["prompt_template"] = node.prompt
                if node.temperature is not None:
                    enriched["temperature"] = node.temperature
                if node.output:
                    enriched["output_schema"] = node.output.model_dump() if hasattr(node.output, "model_dump") else node.output

    return enriched


def _build_execution_graph(workflow_spec: Optional[WorkflowSpec]) -> Dict[str, Any]:
    """
    Build execution graph structure from workflow spec.

    Args:
        workflow_spec: Workflow specification

    Returns:
        Graph structure with nodes and edges
    """
    if not workflow_spec:
        return {"nodes": [], "edges": []}

    nodes = []
    edges = []

    def collect_nodes(node: Node):
        """Recursively collect all nodes from the workflow spec."""
        node_id = node.id
        node_type = node.type if hasattr(node, "type") else "unknown"

        # Build label
        label = node_id
        if isinstance(node, ToolNode):
            label = f"{node_id} ({node.tool})"
        elif isinstance(node, LLMNode):
            label = f"{node_id} (LLM)"

        nodes.append({
            "id": node_id,
            "type": node_type,
            "label": label,
        })

        # Process children for composite nodes
        if isinstance(node, IfNode):
            for child in node.then:
                collect_nodes(child)
            for child in node.else_:
                collect_nodes(child)
        elif isinstance(node, ForEachNode):
            for child in node.body:
                collect_nodes(child)

    # Collect all nodes
    for node in workflow_spec.nodes:
        collect_nodes(node)

    # Get edges from reactflow_graph in meta (contains actual graph connections)
    rf_graph = workflow_spec.meta.get("reactflow_graph") if workflow_spec.meta else None
    if rf_graph and isinstance(rf_graph, dict) and "edges" in rf_graph:
        for edge in rf_graph["edges"]:
            if isinstance(edge, dict) and "source" in edge and "target" in edge:
                edges.append({
                    "source": edge["source"],
                    "target": edge["target"],
                })

    return {"nodes": nodes, "edges": edges}


# ===== Phase 1 Helper Functions =====


def _parse_workflow_spec(spec_dict: Dict[str, Any]) -> Optional[WorkflowSpec]:
    """Parse workflow spec with error suppression."""
    try:
        return WorkflowSpec.model_validate(spec_dict)
    except Exception:
        return None


async def _log_checkpoint_diagnostics(
    checkpointer,
    config: Dict[str, Any],
    run_id: str,
    logger,
) -> None:
    """Log diagnostic information about available checkpoints."""
    try:
        checkpoints_list = []
        async for checkpoint_tuple in checkpointer.alist(config):
            checkpoints_list.append({
                "checkpoint_id": checkpoint_tuple.config.get("configurable", {}).get("checkpoint_id"),
                "checkpoint_ns": checkpoint_tuple.config.get("configurable", {}).get("checkpoint_ns"),
                "thread_id": checkpoint_tuple.config.get("configurable", {}).get("thread_id"),
            })
        if checkpoints_list:
            logger.info(
                f"Found {len(checkpoints_list)} checkpoint(s) for thread_id '{config.get('configurable', {}).get('thread_id')}'",
                extra={"run_id": run_id, "checkpoints": checkpoints_list}
            )
        else:
            logger.warning(
                f"No checkpoints found in alist for thread_id '{config.get('configurable', {}).get('thread_id')}'",
                extra={"run_id": run_id, "config": config}
            )
    except Exception as list_exc:
        logger.warning(
            f"Error listing checkpoints for diagnostic: {list_exc}",
            extra={"run_id": run_id, "config": config}
        )


async def _retrieve_checkpoint(
    checkpointer,
    config: Dict[str, Any],
    run_id: str,
    logger,
):
    """
    Retrieve checkpoint tuple with fallback strategies.

    Tries:
    1. Direct checkpoint retrieval with provided config
    2. Retry with explicit checkpoint_ns=""
    3. Diagnostic logging if no checkpoints found

    Returns:
        Checkpoint tuple if found, None otherwise
    Raises:
        HTTPException with RUN_PROBLEM if checkpoint retrieval fails
    """
    try:
        state_tuple = await checkpointer.aget_tuple(config)
        if state_tuple:
            checkpoint_config = state_tuple.config.get("configurable", {})
            logger.info(
                f"Checkpoint found for run '{run_id}': checkpoint_id={checkpoint_config.get('checkpoint_id')}, checkpoint_ns={checkpoint_config.get('checkpoint_ns')}",
                extra={
                    "run_id": run_id,
                    "checkpoint_id": checkpoint_config.get("checkpoint_id"),
                    "checkpoint_ns": checkpoint_config.get("checkpoint_ns"),
                    "thread_id": checkpoint_config.get("thread_id"),
                }
            )
        else:
            logger.warning(
                f"No checkpoint found for run '{run_id}' with config: {config}",
                extra={"run_id": run_id, "config": config}
            )
    except Exception as checkpoint_exc:
        logger.error(
            f"Failed to retrieve checkpoint for run '{run_id}': {checkpoint_exc}",
            exc_info=True,
            extra={"run_id": run_id}
        )
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Failed to retrieve checkpoint",
            detail=f"Unable to retrieve checkpoint data for run '{run_id}': {str(checkpoint_exc)}",
            status=500,
        )

    if not state_tuple:
        # Try with explicit checkpoint_ns="" in case checkpoints were saved with namespace
        config_with_ns = dict(config)
        config_with_ns.setdefault("configurable", {})["checkpoint_ns"] = ""
        logger.info(
            f"Retrying checkpoint retrieval with explicit checkpoint_ns='' for run '{run_id}'",
            extra={"run_id": run_id, "config_with_ns": config_with_ns}
        )
        try:
            state_tuple = await checkpointer.aget_tuple(config_with_ns)
            if state_tuple:
                logger.info(
                    f"Checkpoint found with checkpoint_ns='' for run '{run_id}'",
                    extra={"run_id": run_id}
                )
                # Update config for subsequent operations
                config.update(config_with_ns)
        except Exception as ns_exc:
            logger.warning(
                f"Retry with checkpoint_ns='' also failed: {ns_exc}",
                extra={"run_id": run_id, "config_with_ns": config_with_ns}
            )

        if not state_tuple:
            _raise_problem(
                type_uri=RUN_PROBLEM,
                title="Run history not found",
                detail=f"No checkpoints found for run '{run_id}'. Checked with config: {config}",
                status=404,
            )

    return state_tuple


async def _extract_node_traces_from_state(
    compiled,
    config: Dict[str, Any],
    workflow_spec: Optional[WorkflowSpec],
    run_id: str,
    logger,
) -> List[Dict[str, Any]]:
    """
    Extract node execution traces from compiled graph state.

    Accesses full state via compiled.workflow.graph.aget_state() and
    extracts all _trace_* keys, enriching them with workflow spec metadata.

    Returns:
        List of node trace dictionaries with enrichment
    """
    nodes = []
    trace_keys_found = set()

    # Get the full state from the graph (includes all keys, not just channel_values)
    graph_state = await compiled.workflow.graph.aget_state(config)

    if graph_state and graph_state.values:
        state_values = graph_state.values
        logger.info(
            f"Retrieved full state from graph with {len(state_values)} keys",
            extra={"run_id": run_id, "state_keys": list(state_values.keys())[:20]}  # Log first 20 keys
        )

        # Extract trace keys from full state
        for key, value in state_values.items():
            if key.startswith("_trace_"):
                node_id = key.replace("_trace_", "")
                if node_id not in trace_keys_found:
                    trace_keys_found.add(node_id)
                    if isinstance(value, dict):
                        node_trace = {
                            "node_id": node_id,
                            "node_type": value.get("node_type", "unknown"),
                            "inputs": value.get("inputs", {}),
                            "output": value.get("output"),
                            "timestamp": value.get("timestamp"),
                            "output_key": value.get("output_key"),
                        }
                        # Enrich with workflow spec metadata
                        try:
                            enriched_node = _enrich_node_with_spec(node_trace, node_id, workflow_spec)
                            nodes.append(enriched_node)
                            logger.info(
                                f"Found trace data for node '{node_id}' in graph state",
                                extra={"run_id": run_id, "node_id": node_id}
                            )
                        except Exception as enrich_exc:
                            logger.warning(
                                f"Failed to enrich node '{node_id}' with spec metadata: {enrich_exc}",
                                exc_info=True,
                                extra={"run_id": run_id, "node_id": node_id}
                            )
                            # Still add the node trace without enrichment
                            nodes.append(node_trace)
    else:
        logger.warning(
            f"No state values found from graph.aget_state() for run '{run_id}'",
            extra={"run_id": run_id}
        )

    return nodes


async def _extract_node_traces_from_checkpoints(
    checkpointer,
    config: Dict[str, Any],
    workflow_spec: Optional[WorkflowSpec],
    run_id: str,
    logger,
) -> List[Dict[str, Any]]:
    """
    Fallback: Extract node traces by iterating checkpoints.

    Used when graph state access fails. Iterates checkpoint channel_values
    to find _trace_* keys.

    Returns:
        List of node trace dictionaries with enrichment
    """
    nodes = []
    trace_keys_found = set()

    try:
        async for checkpoint_tuple in checkpointer.alist(config):
            checkpoint = checkpoint_tuple.checkpoint
            channel_values = checkpoint.get("channel_values", {})

            for key, value in channel_values.items():
                if key.startswith("_trace_"):
                    node_id = key.replace("_trace_", "")
                    if node_id not in trace_keys_found:
                        trace_keys_found.add(node_id)
                        if isinstance(value, dict):
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
                                    f"Failed to enrich node '{node_id}' with spec metadata: {enrich_exc}",
                                    exc_info=True,
                                    extra={"run_id": run_id, "node_id": node_id}
                                )
                                # Still add the node trace without enrichment
                                nodes.append(node_trace)
    except Exception as list_exc:
        logger.error(
            f"Error in fallback checkpoint iteration: {list_exc}",
            exc_info=True,
            extra={"run_id": run_id}
        )

    return nodes


def _serialize_run_timestamps(run: WorkflowRun, logger) -> Dict[str, Optional[str]]:
    """
    Safely serialize run timestamps to ISO format strings.

    Handles AttributeError and ValueError with logging.

    Returns:
        Dictionary with keys: created_at, started_at, finished_at
        Values are ISO strings or None
    """
    created_at_str = None
    started_at_str = None
    finished_at_str = None

    try:
        if run.created_at:
            created_at_str = run.created_at.isoformat()
    except (AttributeError, ValueError) as e:
        logger.warning(
            f"Error serializing created_at for run '{run.run_id}': {e}",
            extra={"run_id": run.run_id}
        )

    try:
        if run.started_at:
            started_at_str = run.started_at.isoformat()
    except (AttributeError, ValueError) as e:
        logger.warning(
            f"Error serializing started_at for run '{run.run_id}': {e}",
            extra={"run_id": run.run_id}
        )

    try:
        if run.finished_at:
            finished_at_str = run.finished_at.isoformat()
    except (AttributeError, ValueError) as e:
        logger.warning(
            f"Error serializing finished_at for run '{run.run_id}': {e}",
            extra={"run_id": run.run_id}
        )

    return {
        "created_at": created_at_str,
        "started_at": started_at_str,
        "finished_at": finished_at_str,
    }


def _build_run_history_response(
    run: WorkflowRun,
    nodes: List[Dict[str, Any]],
    execution_graph: Dict[str, Any],
    timestamps: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    """
    Build the history response structure.

    Combines run metadata, nodes, execution graph, and timestamps
    into the expected response format.

    Returns:
        List containing single history entry dictionary
    """
    # Safe access to workflow_id
    workflow_id = None
    if run.workflow:
        try:
            workflow_id = run.workflow.workflow_id
        except AttributeError:
            pass

    return [{
        "run_id": run.run_id,
        "workflow_id": workflow_id,
        "status": run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
        "created_at": timestamps["created_at"],
        "started_at": timestamps["started_at"],
        "finished_at": timestamps["finished_at"],
        "nodes": nodes,
        "execution_graph": execution_graph,
    }]


# ===== Helper Functions =====


async def _get_run(user: User, run_id: str, session: AsyncSession) -> WorkflowRun:
    """
    Get a workflow run by ID, prefetching the workflow relationship.

    Prefetches the workflow ForeignKey to avoid QuerySet issues when accessing
    run.workflow.workflow_id later.
    """
    try:
        pk = parse_run_public_id(run_id)
    except ValueError:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid run id",
            detail="Run id is invalid",
            status=400,
        )
    stmt = select(WorkflowRun).where(WorkflowRun.id == pk, WorkflowRun.user_id == user.id).options(selectinload(WorkflowRun.workflow))
    result = await session.execute(stmt)
    run = result.scalar_one_or_none()
    if not run:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Run not found",
            detail=f"Run '{run_id}' not found",
            status=404,
        )
    return run


def _serialize_run(run: WorkflowRun) -> api_models.RunResponse:
    """Serialize workflow run to API response."""
    workflow_public_id = (
        make_workflow_public_id(run.workflow_id) if run.workflow_id else None
    )
    return api_models.RunResponse(
        run_id=run.run_id,
        status=run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
        workflow_id=workflow_public_id,
        workflow_version_id=run.workflow_version_id,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        progress=None,
        current_node_id=None,
        last_error=run.error,
    )


# ===== Public API Functions =====


async def get_run_history(user: User, run_id: str) -> api_models.RunHistoryResponse:
    """
    Get workflow execution history with node-based traces.

    Returns node execution traces extracted from the latest checkpoint,
    enriched with workflow spec metadata.
    """
    # Validation: Check database configuration
    from shared.config import config as shared_config
    if not shared_config.DATABASE_URL:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="History unavailable",
            detail="LangGraph checkpointer is not configured",
            status=503,
        )

    # Load run and checkpointer
    async with async_session_maker() as session:
        run = await _get_run(user, run_id, session)
    checkpointer = await get_checkpointer()
    if checkpointer is None:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="History unavailable",
            detail="LangGraph checkpointer failed to initialize",
            status=503,
        )

    # Parse workflow spec (with graceful degradation)
    workflow_spec = _parse_workflow_spec(run.spec)

    # Build run config
    config = _build_run_config(run, run.config)

    logger.info(
        f"Retrieving checkpoint for run '{run.run_id}' with config: {config}",
        extra={
            "run_id": run.run_id,
            "config": config,
            "run_config": run.config,
            "run_status": run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
        }
    )

    # Diagnostic: Log available checkpoints
    await _log_checkpoint_diagnostics(checkpointer, config, run.run_id, logger)

    try:
        # Retrieve checkpoint with fallback logic
        state_tuple = await _retrieve_checkpoint(checkpointer, config, run.run_id, logger)

        # Extract node traces (try graph state first, fall back to checkpoint iteration)
        logger.info(
            f"Compiling workflow to access full state for thread_id '{config.get('configurable', {}).get('thread_id')}'",
            extra={"run_id": run.run_id}
        )

        try:
            # Compile workflow and extract from graph state
            compiled = await compiler.compile(user, run.spec, checkpointer=checkpointer)
            nodes = await _extract_node_traces_from_state(compiled, config, workflow_spec, run.run_id, logger)
        except Exception as graph_exc:
            # Fallback: Extract from checkpoint iteration
            logger.warning(
                f"Error accessing graph state, falling back to checkpoint channel_values: {graph_exc}",
                exc_info=True,
                extra={"run_id": run.run_id}
            )
            nodes = await _extract_node_traces_from_checkpoints(checkpointer, config, workflow_spec, run.run_id, logger)

        # Log summary
        logger.info(
            f"Found {len(nodes)} node trace(s) for run '{run.run_id}'",
            extra={"run_id": run.run_id, "node_count": len(nodes), "node_ids": [n.get("node_id") for n in nodes]}
        )

        # Build execution graph
        execution_graph = _build_execution_graph(workflow_spec)

        # Serialize timestamps
        timestamps = _serialize_run_timestamps(run, logger)

        # Build response
        history = _build_run_history_response(run, nodes, execution_graph, timestamps)

    except HTTPException:
        # Re-raise HTTP exceptions (like 404) without modification
        raise
    except Exception as exc:  # pragma: no cover - bubble up as HTTP problem
        logger.error(
            f"Unexpected error loading run history for run '{run.run_id}': {exc}",
            exc_info=True,
            extra={"run_id": run.run_id, "user_id": user.id}
        )
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Failed to load run history",
            detail=f"An error occurred while loading history for run '{run.run_id}': {str(exc)}",
            status=500,
        )

    return api_models.RunHistoryResponse(run_id=run.run_id, history=history)


async def get_run_status(user: User, run_id: str) -> api_models.RunResponse:
    """Get current status of a workflow run."""
    async with async_session_maker() as session:
        run = await _get_run(user, run_id, session)
        return _serialize_run(run)


async def get_run_result(
    user: User,
    run_id: str,
    *,
    include_state: bool = False,
) -> api_models.RunResultResponse:
    """Get result output from a completed workflow run."""
    async with async_session_maker() as session:
        run = await _get_run(user, run_id, session)
        output = run.output or {}
        workflow_public_id = (
            make_workflow_public_id(run.workflow_id) if run.workflow_id else None
        )
        return api_models.RunResultResponse(
            run_id=run.run_id,
            status=run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
            workflow_id=workflow_public_id,
            workflow_version_id=run.workflow_version_id,
            output=output,
            state=None,
            metrics=run.metrics,
        )


async def list_run_steps(*_: Any, **__: Any) -> None:
    """Placeholder: Run step telemetry not implemented."""
    _raise_problem(
        type_uri=RUN_PROBLEM,
        title="Not implemented",
        detail="Run step telemetry is not implemented",
        status=501,
    )


async def cancel_run(*_: Any, **__: Any) -> None:
    """Placeholder: Run cancellation not implemented."""
    _raise_problem(
        type_uri=RUN_PROBLEM,
        title="Not implemented",
        detail="Run cancellation is not implemented",
        status=501,
    )
