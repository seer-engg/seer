"""Workflow run history, checkpoints, and result retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from api.workflows import models as api_models
from api.workflows.services.shared import (
    RUN_PROBLEM,
    VALIDATION_PROBLEM,
    _build_run_config,
    _raise_problem,
)
from api.agents.checkpointer import get_checkpointer
from shared.config import config as shared_config
from shared.database.workflow_models import (
    User,
    WorkflowRun,
    WorkflowRunStatus,
    parse_run_public_id,
)
from shared.logger import get_logger
from workflow_compiler.runtime.global_compiler import WorkflowCompilerSingleton
from workflow_compiler.schema.models import (
    ForEachNode,
    IfNode,
    LLMNode,
    Node,
    ToolNode,
    WorkflowSpec,
)

logger = get_logger(__name__)
COMPILER = WorkflowCompilerSingleton.instance()

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


def _enrich_node_with_spec(
    node_trace: Dict[str, Any],
    node_id: str,
    workflow_spec: Optional[WorkflowSpec],
) -> Dict[str, Any]:
    """Enrich node trace with workflow spec metadata."""
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
    """Build execution graph structure from workflow spec."""
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



async def _get_run(user: User, run_id: str) -> WorkflowRun:
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
    # Use filter().prefetch_related().first() instead of .get() to prefetch workflow
    run = await WorkflowRun.filter(id=pk, user=user).prefetch_related('workflow').first()
    if not run:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="Run not found",
            detail=f"Run '{run_id}' not found",
            status=404,
        )
    return run


async def get_run_history(user: User, run_id: str) -> api_models.RunHistoryResponse:
    # pylint: disable=too-many-statements, too-many-nested-blocks
    """
    Get workflow execution history with node-based traces.

    Returns node execution traces extracted from the latest checkpoint,
    enriched with workflow spec metadata.
    """
    if not shared_config.DATABASE_URL:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="History unavailable",
            detail="LangGraph checkpointer is not configured",
            status=503,
        )
    run = await _get_run(user, run_id)
    checkpointer = await get_checkpointer()
    if checkpointer is None:
        _raise_problem(
            type_uri=RUN_PROBLEM,
            title="History unavailable",
            detail="LangGraph checkpointer failed to initialize",
            status=503,
        )

    # Parse workflow spec for node enrichment
    workflow_spec: Optional[WorkflowSpec] = None
    try:
        workflow_spec = WorkflowSpec.model_validate(run.spec)
    except Exception:
        # If spec parsing fails, continue without enrichment
        pass

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
        },
    )

    # Diagnostic: Try to list checkpoints for this thread_id to see if any exist
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
                "Found %s checkpoint(s) for thread_id '%s'",
                len(checkpoints_list),
                config.get("configurable", {}).get("thread_id"),
                extra={"run_id": run.run_id, "checkpoints": checkpoints_list},
            )
        else:
            logger.warning(
                "No checkpoints found in alist for thread_id '%s'",
                config.get("configurable", {}).get("thread_id"),
                extra={"run_id": run.run_id, "config": config},
            )
    except Exception as list_exc:
        logger.warning(
            "Error listing checkpoints for diagnostic: %s",
            list_exc,
            extra={"run_id": run.run_id, "config": config},
        )

    try:
        # Get latest checkpoint only (not all historical checkpoints)
        try:
            state_tuple = await checkpointer.aget_tuple(config)
            if state_tuple:
                checkpoint_config = state_tuple.config.get("configurable", {})
                logger.info(
                    "Checkpoint found for run '%s': checkpoint_id=%s, checkpoint_ns=%s",
                    run.run_id,
                    checkpoint_config.get("checkpoint_id"),
                    checkpoint_config.get("checkpoint_ns"),
                    extra={
                        "run_id": run.run_id,
                        "checkpoint_id": checkpoint_config.get("checkpoint_id"),
                        "checkpoint_ns": checkpoint_config.get("checkpoint_ns"),
                        "thread_id": checkpoint_config.get("thread_id"),
                    },
                )
            else:
                logger.warning(
                    "No checkpoint found for run '%s' with config: %s",
                    run.run_id,
                    config,
                    extra={"run_id": run.run_id, "config": config, "run_config": run.config},
                )
        except Exception as checkpoint_exc:
            logger.error(
                "Failed to retrieve checkpoint for run '%s': %s",
                run.run_id,
                checkpoint_exc,
                exc_info=True,
                extra={"run_id": run.run_id, "user_id": user.id}
            )
            _raise_problem(
                type_uri=RUN_PROBLEM,
                title="Failed to retrieve checkpoint",
                detail=f"Unable to retrieve checkpoint data for run '{run.run_id}': {str(checkpoint_exc)}",
                status=500,
            )

        if not state_tuple:
            # Try with explicit checkpoint_ns="" in case checkpoints were saved with namespace
            config_with_ns = dict(config)
            config_with_ns.setdefault("configurable", {})["checkpoint_ns"] = ""
            logger.info(
                "Retrying checkpoint retrieval with explicit checkpoint_ns='' for run '%s'",
                run.run_id,
                extra={"run_id": run.run_id, "config_with_ns": config_with_ns}
            )
            try:
                state_tuple = await checkpointer.aget_tuple(config_with_ns)
                if state_tuple:
                    logger.info(
                        "Checkpoint found with checkpoint_ns='' for run '%s'",
                        run.run_id,
                        extra={"run_id": run.run_id}
                    )
                    # Use the config with namespace for subsequent operations
                    config = config_with_ns
            except Exception as ns_exc:
                logger.warning(
                    "Retry with checkpoint_ns='' also failed: %s",
                    ns_exc,
                    extra={"run_id": run.run_id, "config_with_ns": config_with_ns}
                )

            if not state_tuple:
                _raise_problem(
                    type_uri=RUN_PROBLEM,
                    title="Run history not found",
                    detail=f"No checkpoints found for run '{run.run_id}'. Checked with config: {config}",
                    status=404,
                )

        # Extract node traces from the compiled graph's state
        # LangGraph stores full state but only exposes certain keys in channel_values
        # Using graph.aget_state() gives us access to the complete state including trace keys
        nodes = []
        trace_keys_found = set()

        logger.info(
            "Compiling workflow to access full state for thread_id '%s'",
            config.get("configurable", {}).get("thread_id"),
            extra={"run_id": run.run_id}
        )

        try:
            # Compile the workflow to get access to the graph
            compiled = await COMPILER.compile(
                user,
                run.spec,
                checkpointer=checkpointer,
            )

            # Get the full state from the graph (includes all keys, not just channel_values)
            # compiled is UserBoundCompiledWorkflow, which has a workflow attribute
            graph_state = await compiled.workflow.graph.aget_state(config)

            if graph_state and graph_state.values:
                state_values = graph_state.values
                logger.info(
                    "Retrieved full state from graph with %s keys",
                    len(state_values),
                    extra={"run_id": run.run_id, "state_keys": list(state_values.keys())[:20]}  # Log first 20 keys
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
                                    # Still add the node trace without enrichment
                                    nodes.append(node_trace)
            else:
                logger.warning(
                    "No state values found from graph.aget_state() for run '%s'",
                    run.run_id,
                    extra={"run_id": run.run_id}
                )

        except Exception as graph_exc:
            logger.warning(
                "Error accessing graph state, falling back to checkpoint channel_values: %s",
                graph_exc,
                exc_info=True,
                extra={"run_id": run.run_id}
            )
            # Fallback to checking channel_values from checkpoints
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
                                            "Failed to enrich node '%s' with spec metadata: %s",
                                            node_id,
                                            enrich_exc,
                                            exc_info=True,
                                            extra={"run_id": run.run_id, "node_id": node_id}
                                        )
                                        # Still add the node trace without enrichment
                                        nodes.append(node_trace)
            except Exception as list_exc:
                logger.error(
                    "Error in fallback checkpoint iteration: %s",
                    list_exc,
                    exc_info=True,
                    extra={"run_id": run.run_id}
                )

        # Log summary of trace data found
        logger.info(
            "Found %s node trace(s) for run '%s'",
            len(nodes),
            run.run_id,
            extra={"run_id": run.run_id, "node_count": len(nodes), "node_ids": [n.get("node_id") for n in nodes]}
        )

        # Build execution graph from workflow spec
        execution_graph = _build_execution_graph(workflow_spec)

        # Safe access to workflow_id
        workflow_id = None
        if run.workflow:
            try:
                workflow_id = run.workflow.workflow_id
            except AttributeError as e:
                logger.warning(
                    "Error accessing workflow_id for run '%s': %s",
                    run.run_id,
                    e,
                    extra={"run_id": run.run_id}
                )

        # Safe datetime serialization
        created_at_str = None
        started_at_str = None
        finished_at_str = None

        try:
            if run.created_at:
                created_at_str = run.created_at.isoformat()
        except (AttributeError, ValueError) as e:
            logger.warning(
                "Error serializing created_at for run '%s': %s",
                run.run_id,
                e,
                extra={"run_id": run.run_id}
            )

        try:
            if run.started_at:
                started_at_str = run.started_at.isoformat()
        except (AttributeError, ValueError) as e:
            logger.warning(
                "Error serializing started_at for run '%s': %s",
                run.run_id,
                e,
                extra={"run_id": run.run_id}
            )

        try:
            if run.finished_at:
                finished_at_str = run.finished_at.isoformat()
        except (AttributeError, ValueError) as e:
            logger.warning(
                "Error serializing finished_at for run '%s': %s",
                run.run_id,
                e,
                extra={"run_id": run.run_id}
            )

        # Return node-based structure
        # The history field contains a single entry with node-based data
        history = [{
            "run_id": run.run_id,
            "workflow_id": workflow_id,
            "status": run.status.value if isinstance(run.status, WorkflowRunStatus) else run.status,
            "created_at": created_at_str,
            "started_at": started_at_str,
            "finished_at": finished_at_str,
            "nodes": nodes,
            "execution_graph": execution_graph,
        }]

    except HTTPException:
        # Re-raise HTTP exceptions (like 404) without modification
        raise
    except Exception as exc:  # pragma: no cover - bubble up as HTTP problem
        logger.error(
            "Unexpected error loading run history for run '%s': %s",
            run.run_id,
            exc,
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
    run = await _get_run(user, run_id)
    from api.workflows.services.execution import _serialize_run  # pylint: disable=import-outside-toplevel
    return _serialize_run(run)
