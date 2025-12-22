"""
Workflow service layer for business logic.
"""
from typing import List, Optional
from datetime import datetime

from fastapi import HTTPException
from shared.logger import get_logger

from .models import (
    Workflow,
    WorkflowBlock,
    WorkflowEdge,
    WorkflowExecution,
    BlockExecution,
    WorkflowCreate,
    WorkflowUpdate,
)
from .schema import validate_workflow_graph

logger = get_logger("api.workflows.services")


async def create_workflow(
    user_id: str,
    payload: WorkflowCreate,
) -> Workflow:
    """
    Create a new workflow.
    
    Args:
        user_id: User ID from auth
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
            user_id=user_id,
            graph_data=payload.graph_data,
            schema_version=payload.schema_version,
            is_active=payload.is_active,
        )
        
        # Create blocks and edges
        await _sync_workflow_blocks_and_edges(workflow, payload.graph_data)
        
        logger.info(f"Created workflow {workflow.id} for user {user_id}")
        return workflow
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create workflow")


async def get_workflow(workflow_id: int, user_id: str) -> Workflow:
    """
    Get a workflow by ID.
    
    Args:
        workflow_id: Workflow ID
        user_id: User ID for authorization
        
    Returns:
        Workflow model
        
    Raises:
        HTTPException: If workflow not found or unauthorized
    """
    workflow = await Workflow.get_or_none(id=workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if workflow.user_id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    return workflow


async def list_workflows(user_id: str) -> List[Workflow]:
    """
    List all workflows for a user.
    
    Args:
        user_id: User ID
        
    Returns:
        List of workflows
    """
    workflows = await Workflow.filter(user_id=user_id, is_active=True).all()
    return workflows


async def update_workflow(
    workflow_id: int,
    user_id: str,
    payload: WorkflowUpdate,
) -> Workflow:
    """
    Update a workflow.
    
    Args:
        workflow_id: Workflow ID
        user_id: User ID for authorization
        payload: Update payload
        
    Returns:
        Updated workflow
        
    Raises:
        HTTPException: If workflow not found or unauthorized
    """
    workflow = await get_workflow(workflow_id, user_id)
    
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
    
    logger.info(f"Updated workflow {workflow_id} for user {user_id}")
    return workflow


async def delete_workflow(workflow_id: int, user_id: str) -> None:
    """
    Delete a workflow (soft delete).
    
    Args:
        workflow_id: Workflow ID
        user_id: User ID for authorization
        
    Raises:
        HTTPException: If workflow not found or unauthorized
    """
    workflow = await get_workflow(workflow_id, user_id)
    
    workflow.is_active = False
    workflow.updated_at = datetime.utcnow()
    await workflow.save()
    
    logger.info(f"Deleted workflow {workflow_id} for user {user_id}")


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
    user_id: str,
    input_data: Optional[dict] = None,
) -> WorkflowExecution:
    """
    Create a workflow execution record.
    
    Args:
        workflow_id: Workflow ID
        user_id: User ID
        input_data: Input data for execution
        
    Returns:
        Created execution record
    """
    workflow = await get_workflow(workflow_id, user_id)
    
    execution = await WorkflowExecution.create(
        workflow=workflow,
        user_id=user_id,
        status='running',
        input_data=input_data,
    )
    
    return execution


async def update_execution(
    execution_id: int,
    user_id: str,
    status: str,
    output_data: Optional[dict] = None,
    error_message: Optional[str] = None,
) -> WorkflowExecution:
    """
    Update workflow execution status.
    
    Args:
        execution_id: Execution ID
        user_id: User ID for authorization
        status: New status
        output_data: Output data
        error_message: Error message if failed
        
    Returns:
        Updated execution
    """
    execution = await WorkflowExecution.get_or_none(id=execution_id)
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    if execution.user_id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    execution.status = status
    execution.output_data = output_data
    execution.error_message = error_message
    
    if status in ('completed', 'failed'):
        execution.completed_at = datetime.utcnow()
    
    await execution.save()
    return execution


async def list_executions(
    workflow_id: int,
    user_id: str,
    limit: int = 50,
) -> List[WorkflowExecution]:
    """
    List executions for a workflow.
    
    Args:
        workflow_id: Workflow ID
        user_id: User ID for authorization
        limit: Maximum number of executions to return
        
    Returns:
        List of executions
    """
    workflow = await get_workflow(workflow_id, user_id)
    
    executions = await WorkflowExecution.filter(
        workflow=workflow,
        user_id=user_id,
    ).order_by('-started_at').limit(limit).all()
    
    return executions


__all__ = [
    "create_workflow",
    "get_workflow",
    "list_workflows",
    "update_workflow",
    "delete_workflow",
    "create_execution",
    "update_execution",
    "list_executions",
]

