"""
Workflow service layer for business logic.
"""
from typing import List, Optional
from datetime import datetime

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
    WorkflowCreate,
    WorkflowUpdate,
)
from .schema import validate_workflow_graph

logger = get_logger("api.workflows.services")


async def create_workflow(
    user_id: Optional[str],
    payload: WorkflowCreate,
) -> Workflow:
    """
    Create a new workflow.
    
    Args:
        user_id: User ID from auth (None in self-hosted mode)
        payload: Workflow creation payload
        
    Returns:
        Created workflow
    """
    try:
        # Validate workflow graph
        validate_workflow_graph(payload.graph_data)
        
        # In self-hosted mode, user_id is None
        # In cloud mode, user_id is required
        if config.is_cloud_mode and not user_id:
            raise HTTPException(status_code=401, detail="Authentication required in cloud mode")
        
        # Create workflow
        workflow = await Workflow.create(
            name=payload.name,
            description=payload.description,
            user_id=user_id,  # None in self-hosted, set in cloud
            graph_data=payload.graph_data,
            schema_version=payload.schema_version,
            is_active=payload.is_active,
        )
        
        # Create blocks and edges
        await _sync_workflow_blocks_and_edges(workflow, payload.graph_data)
        
        logger.info(f"Created workflow {workflow.id} for user {user_id or 'self-hosted'}")
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
            block.block_type = node.get('type', 'code')
            block.block_config = data.get('config', {})
            block.python_code = data.get('python_code')
            block.oauth_scope = data.get('oauth_scope')
            block.position_x = position.get('x', 0)
            block.position_y = position.get('y', 0)
            await block.save()
        else:
            await WorkflowBlock.create(
                workflow=workflow,
                block_id=block_id,
                block_type=node.get('type', 'code'),
                block_config=data.get('config', {}),
                python_code=data.get('python_code'),
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
                source_handle=edge_data.get('sourceHandle'),
                target_handle=edge_data.get('targetHandle'),
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
    
    session = await WorkflowChatSession.get_or_none(
        id=session_id,
        workflow=workflow,
    )
    
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
    
    session = await WorkflowChatSession.get_or_none(
        thread_id=thread_id,
        workflow=workflow,
    )
    
    if not session:
        return None
    
    return session


async def list_chat_sessions(
    workflow_id: int,
    user: User,
    limit: int = 50,
) -> List[WorkflowChatSession]:
    """
    List chat sessions for a workflow.
    
    Args:
        workflow_id: Workflow ID
        user: User
        limit: Maximum number of sessions to return
        
    Returns:
        List of chat sessions
    """
    workflow = await get_workflow(workflow_id)
    
    sessions = await WorkflowChatSession.filter(
        workflow=workflow,
        user=user,
    ).order_by('-updated_at').limit(limit).all()
    
    return sessions


async def save_chat_message(
    session_id: int,
    role: str,
    content: str,
    thinking: Optional[str] = None,
    suggested_edits: Optional[dict] = None,
    metadata: Optional[dict] = None,
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
    ).order_by('created_at').all()
    
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
]

