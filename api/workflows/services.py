"""
Workflow service layer for business logic.
"""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from copy import deepcopy

from fastapi import HTTPException
from shared.logger import get_logger
from shared.config import config
from shared.database.models import User
from .models import (
    Workflow,
    WorkflowBlock,
    WorkflowEdge,
    WorkflowExecution,
    BlockExecution,
    WorkflowChatSession,
    WorkflowChatMessage,
    WorkflowProposal,
    WorkflowCreate,
    WorkflowUpdate,
)
from .schema import validate_workflow_graph

logger = get_logger("api.workflows.services")


async def create_workflow(
    user: User,
    payload: WorkflowCreate,
) -> Workflow:
    """
    Create a new workflow.
    
    Args:
        user: User
        payload: Workflow creation payload
        
    Returns:
        Created workflow
    """
    try:
        # Validate workflow graph
        validate_workflow_graph(payload.graph_data)
        
        # Create workflow
        workflow = await Workflow.create(
            name=payload.name,
            description=payload.description,
            user=user,
            graph_data=payload.graph_data,
            schema_version=payload.schema_version,
            is_active=payload.is_active,
        )
        
        # Create blocks and edges
        await _sync_workflow_blocks_and_edges(workflow, payload.graph_data)
        
        logger.info(f"Created workflow {workflow.id}")
        return workflow
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create workflow")


async def get_workflow(workflow_id: int) -> Workflow:
    """
    Get a workflow by ID.
    
    Args:
        workflow_id: Workflow ID
        user_id: User ID for authorization (None in self-hosted mode)
        
    Returns:
        Workflow model
        
    Raises:
        HTTPException: If workflow not found or unauthorized
    """
    workflow = await Workflow.get_or_none(id=workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return workflow


async def list_workflows(user: User) -> List[Workflow]:
    """
    List all workflows for a user.
    
    Args:
        user: User
        
    Returns:
        List of workflows
    """
    workflows = await Workflow.filter(user=user, is_active=True).all()
    
    return workflows


async def update_workflow(
    workflow_id: int,
    payload: WorkflowUpdate,
) -> Workflow:
    """
    Update a workflow.
    
    Args:
        workflow_id: Workflow ID
        payload: Update payload
        
    Returns:
        Updated workflow
        
    Raises:
        HTTPException: If workflow not found or unauthorized
    """
    workflow = await get_workflow(workflow_id)
    
    # Validate graph if provided
    if payload.graph_data:
        validate_workflow_graph(payload.graph_data)
    
    # Update fields
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(workflow, field, value)
    
    workflow.updated_at = datetime.utcnow()
    await workflow.save()
    
    # Sync blocks and edges if graph_data was updated
    if payload.graph_data:
        await _sync_workflow_blocks_and_edges(workflow, payload.graph_data)
        # Invalidate cached graph
        from .graph_builder import get_workflow_graph_builder
        builder = await get_workflow_graph_builder()
        builder.invalidate_cache(workflow_id)
    
    logger.info(f"Updated workflow {workflow_id} ")
    return workflow


async def delete_workflow(workflow_id: int) -> None:
    """
    Delete a workflow (soft delete).
    
    Args:
        workflow_id: Workflow ID
        
    Raises:
        HTTPException: If workflow not found or unauthorized
    """
    workflow = await get_workflow(workflow_id)
    
    workflow.is_active = False
    workflow.updated_at = datetime.utcnow()
    await workflow.save()
    
    logger.info(f"Deleted workflow {workflow_id}")


async def _sync_workflow_blocks_and_edges(
    workflow: Workflow,
    graph_data: dict,
) -> None:
    """
    Synchronize workflow blocks and edges from graph data.
    
    Args:
        workflow: Workflow model
        graph_data: ReactFlow graph data
    """
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # Get existing blocks
    existing_blocks = {block.block_id: block for block in await WorkflowBlock.filter(workflow=workflow).all()}
    
    # Create or update blocks
    for node in nodes:
        block_id = node['id']
        data = node.get('data', {})
        position = node.get('position', {})
        
        if block_id in existing_blocks:
            block = existing_blocks[block_id]
            block.block_type = node.get('type', 'tool')
            block.block_config = data.get('config', {})
            block.oauth_scope = data.get('oauth_scope')
            block.position_x = position.get('x', 0)
            block.position_y = position.get('y', 0)
            await block.save()
        else:
            await WorkflowBlock.create(
                workflow=workflow,
                block_id=block_id,
                block_type=node.get('type', 'tool'),
                block_config=data.get('config', {}),
                oauth_scope=data.get('oauth_scope'),
                position_x=position.get('x', 0),
                position_y=position.get('y', 0),
            )
    
    # Delete blocks that are no longer in graph
    node_ids = {node['id'] for node in nodes}
    for block_id, block in existing_blocks.items():
        if block_id not in node_ids:
            await block.delete()
    
    # Delete all existing edges
    await WorkflowEdge.filter(workflow=workflow).delete()
    
    # Create new edges
    existing_blocks_dict = {block.block_id: block for block in await WorkflowBlock.filter(workflow=workflow).all()}
    
    for edge_data in edges:
        source_block = existing_blocks_dict.get(edge_data['source'])
        target_block = existing_blocks_dict.get(edge_data['target'])
        
        if source_block and target_block:
            await WorkflowEdge.create(
                workflow=workflow,
                source_block=source_block,
                target_block=target_block,
                source_handle=None,
                target_handle=None,
            )


async def create_execution(
    workflow_id: int,
    user: User,
    input_data: Optional[dict] = None,
) -> WorkflowExecution:
    """
    Create a workflow execution record.
    
    Args:
        workflow_id: Workflow ID
        user: User
        input_data: Input data for execution
        
    Returns:
        Created execution record
    """
    workflow = await get_workflow(workflow_id)
    
    execution = await WorkflowExecution.create(
        workflow=workflow,
        user=user,
        status='running',
        input_data=input_data,
    )
    
    return execution


async def update_execution(
    execution_id: int,
    status: str,
    output_data: Optional[dict] = None,
    error_message: Optional[str] = None,
) -> WorkflowExecution:
    """
    Update workflow execution status.
    
    Args:
        execution_id: Execution ID
        status: New status
        output_data: Output data
        error_message: Error message if failed
        
    Returns:
        Updated execution
    """
    execution = await WorkflowExecution.get_or_none(id=execution_id)
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution.status = status
    execution.output_data = output_data
    execution.error_message = error_message
    
    if status in ('completed', 'failed'):
        execution.completed_at = datetime.utcnow()
    
    await execution.save()
    return execution


async def list_executions(
    workflow_id: int,
    limit: int = 50,
) -> List[WorkflowExecution]:
    """
    List executions for a workflow.
    
    Args:
        workflow_id: Workflow ID
        user_id: User ID for authorization (None in self-hosted mode)
        limit: Maximum number of executions to return
        
    Returns:
        List of executions
    """
    workflow = await get_workflow(workflow_id)
    
    executions = await WorkflowExecution.filter(
        workflow=workflow,
    ).order_by('-started_at').limit(limit).all()
    
    return executions


async def get_execution(
    execution_id: int,
    workflow_id: int,
) -> WorkflowExecution:
    """
    Get a single execution by ID.
    
    Args:
        execution_id: Execution ID
        workflow_id: Workflow ID (for authorization)
        
    Returns:
        Execution object
        
    Raises:
        HTTPException: If execution not found or unauthorized
    """
    # Verify workflow access first
    workflow = await get_workflow(workflow_id)
    
    # Get execution
    execution = await WorkflowExecution.get_or_none(
        id=execution_id,
        workflow=workflow,
    )
    
    if not execution:
        raise HTTPException(
            status_code=404,
            detail=f"Execution {execution_id} not found"
        )
    
    return execution


# ============================================================================
# Chat Session Services
# ============================================================================

async def create_chat_session(
    workflow_id: int,
    user: User,
    thread_id: str,
    title: Optional[str] = None,
) -> WorkflowChatSession:
    """
    Create a new chat session for a workflow.
    
    Args:
        workflow_id: Workflow ID
        user: User
        thread_id: LangGraph thread ID
        title: Optional session title
        
    Returns:
        Created chat session
    """
    workflow = await get_workflow(workflow_id)
    
    session = await WorkflowChatSession.create(
        workflow=workflow,
        user=user,
        thread_id=thread_id,
        title=title,
    )
    
    # Fetch the user relationship to ensure it's loaded
    await session.fetch_related('user')
    
    logger.info(f"Created chat session {session.id} for workflow {workflow_id}")
    return session


async def get_chat_session(
    session_id: int,
    workflow_id: int,
) -> WorkflowChatSession:
    """
    Get a chat session with its messages.
    
    Args:
        session_id: Session ID
        workflow_id: Workflow ID (for authorization)
        
    Returns:
        Chat session with messages
        
    Raises:
        HTTPException: If session not found or unauthorized
    """
    # Verify workflow access first
    workflow = await get_workflow(workflow_id)
    
    session = await WorkflowChatSession.filter(
        id=session_id,
        workflow=workflow,
    ).prefetch_related('user').first()
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Chat session {session_id} not found"
        )
    
    return session


async def get_chat_session_by_thread_id(
    thread_id: str,
    workflow_id: int,
) -> Optional[WorkflowChatSession]:
    """
    Get a chat session by thread ID.
    
    Args:
        thread_id: LangGraph thread ID
        workflow_id: Workflow ID (for authorization)
        user_id: User ID for authorization (None in self-hosted mode)
        
    Returns:
        Chat session if found, None otherwise
    """
    # Verify workflow access first
    workflow = await get_workflow(workflow_id)
    
    session = await WorkflowChatSession.filter(
        thread_id=thread_id,
        workflow=workflow,
    ).prefetch_related('user').first()
    
    if not session:
        return None
    
    return session


async def list_chat_sessions(
    workflow_id: int,
    user: User,
    limit: int = 50,
    offset: int = 0,
) -> List[WorkflowChatSession]:
    """
    List chat sessions for a workflow.
    
    Args:
        workflow_id: Workflow ID
        user: User
        limit: Maximum number of sessions to return
        offset: Number of sessions to skip
        
    Returns:
        List of chat sessions
    """
    workflow = await get_workflow(workflow_id)
    
    sessions = await WorkflowChatSession.filter(
        workflow=workflow,
        user=user,
    ).prefetch_related('user').order_by('-created_at').offset(offset).limit(limit).all()
    
    return sessions


async def save_chat_message(
    session_id: int,
    role: str,
    content: str,
    thinking: Optional[str] = None,
    suggested_edits: Optional[dict] = None,
    metadata: Optional[dict] = None,
    proposal: Optional[WorkflowProposal] = None,
) -> WorkflowChatMessage:
    """
    Save a chat message to the database.
    
    Args:
        session_id: Session ID
        role: Message role ('user' or 'assistant')
        content: Message content
        thinking: Optional thinking/reasoning steps
        suggested_edits: Optional suggested workflow edits
        metadata: Optional metadata (model used, etc.)
        proposal: Optional proposal linked to this message
        
    Returns:
        Created message
    """
    session = await WorkflowChatSession.get_or_none(id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Update session updated_at timestamp
    session.updated_at = datetime.utcnow()
    await session.save()
    
    message = await WorkflowChatMessage.create(
        session=session,
        proposal=proposal,
        role=role,
        content=content,
        thinking=thinking,
        suggested_edits=suggested_edits,
        metadata=metadata,
    )
    
    logger.debug(f"Saved chat message {message.id} to session {session_id}")
    return message


async def load_chat_history(
    session_id: int,
) -> List[WorkflowChatMessage]:
    """
    Load chat history for a session.
    
    Args:
        session_id: Session ID
        
    Returns:
        List of messages ordered by creation time
    """
    messages = await WorkflowChatMessage.filter(
        session_id=session_id
    ).prefetch_related('proposal__created_by', 'proposal__workflow', 'proposal__session').order_by('created_at').all()
    
    return messages


async def update_chat_session_title(
    session_id: int,
    workflow_id: int,
    user_id: Optional[str],
    title: str,
) -> WorkflowChatSession:
    """
    Update chat session title.
    
    Args:
        session_id: Session ID
        workflow_id: Workflow ID (for authorization)
        user_id: User ID for authorization
        title: New title
        
    Returns:
        Updated session
    """
    session = await get_chat_session(session_id, workflow_id, user_id)
    
    session.title = title
    session.updated_at = datetime.utcnow()
    await session.save()
    
    return session


def _with_default_graph(graph_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a deepcopy of graph data with default nodes/edges."""
    base = graph_data or {}
    return {
        "nodes": deepcopy(base.get("nodes", [])),
        "edges": deepcopy(base.get("edges", [])),
    }


def _apply_patch_ops(
    graph_data: Optional[Dict[str, Any]],
    patch_ops: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply patch operations to a workflow graph."""
    updated_graph = _with_default_graph(graph_data)
    nodes = updated_graph["nodes"]
    edges = updated_graph["edges"]
    
    def _ensure_node_defaults(node: Dict[str, Any]) -> Dict[str, Any]:
        if not node:
            return node
        block_type = node.get("type")
        data = node.get("data") or {}
        config = data.get("config") or {}
        if block_type == "for_loop":
            config.setdefault("array_var", "items")
            config.setdefault("item_var", "item")
        data["config"] = config
        node["data"] = data
        return node
    
    def _find_node_index(node_id: str) -> Optional[int]:
        for idx, node in enumerate(nodes):
            if node.get("id") == node_id:
                return idx
        return None
    
    for op in patch_ops:
        op_type = (op or {}).get("op")
        if not op_type:
            raise HTTPException(status_code=400, detail="Patch operation missing 'op'")
        
        if op_type == "add_node":
            node = op.get("node")
            if not node or "id" not in node:
                raise HTTPException(status_code=400, detail="add_node requires node.id")
            if _find_node_index(node["id"]) is not None:
                raise HTTPException(status_code=400, detail=f"Node '{node['id']}' already exists")
            nodes.append(_ensure_node_defaults(node))
        
        elif op_type == "update_node":
            node = op.get("node")
            node_id = op.get("node_id") or (node or {}).get("id")
            if not node_id or not node:
                raise HTTPException(status_code=400, detail="update_node requires node_id and node")
            idx = _find_node_index(node_id)
            if idx is None:
                raise HTTPException(status_code=400, detail=f"Node '{node_id}' not found")
            nodes[idx] = _ensure_node_defaults(node)
        
        elif op_type == "remove_node":
            node_id = op.get("node_id")
            if not node_id:
                raise HTTPException(status_code=400, detail="remove_node requires node_id")
            if _find_node_index(node_id) is None:
                raise HTTPException(status_code=400, detail=f"Node '{node_id}' not found")
            nodes[:] = [node for node in nodes if node.get("id") != node_id]
            edges[:] = [
                edge for edge in edges
                if edge.get("source") != node_id and edge.get("target") != node_id
            ]
        
        elif op_type == "add_edge":
            edge = op.get("edge")
            if not edge:
                raise HTTPException(status_code=400, detail="add_edge requires edge payload")
            source = edge.get("source")
            target = edge.get("target")
            if not source or not target:
                raise HTTPException(status_code=400, detail="add_edge requires source/target")
            if _find_node_index(source) is None or _find_node_index(target) is None:
                raise HTTPException(status_code=400, detail=f"Edge references unknown nodes {source}->{target}")
            already_exists = any(
                existing.get("source") == source and existing.get("target") == target
                for existing in edges
            )
            if already_exists:
                raise HTTPException(status_code=400, detail=f"Edge {source}->{target} already exists")
            edges.append(edge)
        
        elif op_type == "remove_edge":
            edge = op.get("edge", {})
            edge_id = op.get("edge_id") or edge.get("id")
            source = edge.get("source") or op.get("source_id")
            target = edge.get("target") or op.get("target_id")
            if not edge_id and not (source and target):
                raise HTTPException(status_code=400, detail="remove_edge requires edge_id or source/target")
            edges[:] = [
                existing for existing in edges
                if not (
                    (edge_id and existing.get("id") == edge_id) or
                    (source and target and existing.get("source") == source and existing.get("target") == target)
                )
            ]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown patch operation '{op_type}'")
    
    return updated_graph


def preview_patch_ops(
    graph_data: Optional[Dict[str, Any]],
    patch_ops: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a preview graph after applying patch ops (without persistence)."""
    preview_graph = _apply_patch_ops(graph_data, patch_ops)
    # Validate preview to catch schema issues early
    validate_workflow_graph(preview_graph)
    return preview_graph


async def create_workflow_proposal(
    workflow: Workflow,
    session: Optional[WorkflowChatSession],
    user: User,
    summary: str,
    patch_ops: List[Dict[str, Any]],
    preview_graph: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> WorkflowProposal:
    """Persist a workflow proposal."""
    if not patch_ops:
        raise HTTPException(status_code=400, detail="Proposal requires at least one patch op")
    
    safe_summary = (summary or "").strip() or "Workflow changes"
    if len(safe_summary) > 512:
        safe_summary = f"{safe_summary[:509]}..."
    
    proposal = await WorkflowProposal.create(
        workflow=workflow,
        session=session,
        created_by=user,
        summary=safe_summary,
        patch_ops=patch_ops,
        preview_graph=preview_graph,
        status=WorkflowProposal.STATUS_PENDING,
        metadata=metadata,
    )
    return proposal


async def get_workflow_proposal(
    workflow_id: int,
    proposal_id: int,
) -> WorkflowProposal:
    """Fetch a workflow proposal."""
    workflow = await get_workflow(workflow_id)
    proposal = await WorkflowProposal.get_or_none(id=proposal_id, workflow=workflow)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


async def accept_workflow_proposal(
    workflow_id: int,
    proposal_id: int,
) -> Tuple[WorkflowProposal, Workflow]:
    """Apply workflow proposal and mark accepted."""
    proposal = await get_workflow_proposal(workflow_id, proposal_id)
    if proposal.status != WorkflowProposal.STATUS_PENDING:
        raise HTTPException(status_code=400, detail="Proposal is not pending")
    
    workflow = await proposal.workflow
    updated_graph = _apply_patch_ops(workflow.graph_data, proposal.patch_ops or [])
    validate_workflow_graph(updated_graph)
    
    workflow.graph_data = updated_graph
    workflow.updated_at = datetime.utcnow()
    await workflow.save()
    await _sync_workflow_blocks_and_edges(workflow, updated_graph)
    
    from .graph_builder import get_workflow_graph_builder
    builder = await get_workflow_graph_builder()
    builder.invalidate_cache(workflow.id)
    
    proposal.status = WorkflowProposal.STATUS_ACCEPTED
    proposal.applied_graph = updated_graph
    proposal.decided_at = datetime.utcnow()
    await proposal.save()
    
    return proposal, workflow


async def reject_workflow_proposal(
    workflow_id: int,
    proposal_id: int,
) -> WorkflowProposal:
    """Reject workflow proposal."""
    proposal = await get_workflow_proposal(workflow_id, proposal_id)
    if proposal.status != WorkflowProposal.STATUS_PENDING:
        raise HTTPException(status_code=400, detail="Proposal is not pending")
    
    proposal.status = WorkflowProposal.STATUS_REJECTED
    proposal.decided_at = datetime.utcnow()
    await proposal.save()
    return proposal


__all__ = [
    "create_workflow",
    "get_workflow",
    "list_workflows",
    "update_workflow",
    "delete_workflow",
    "create_execution",
    "update_execution",
    "list_executions",
    "get_execution",
    "create_chat_session",
    "get_chat_session",
    "get_chat_session_by_thread_id",
    "list_chat_sessions",
    "save_chat_message",
    "load_chat_history",
    "update_chat_session_title",
    "preview_patch_ops",
    "create_workflow_proposal",
    "get_workflow_proposal",
    "accept_workflow_proposal",
    "reject_workflow_proposal",
]

