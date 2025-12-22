"""
Workflow API router for CRUD and execution endpoints.
"""
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
from shared.logger import get_logger

from .models import (
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowPublic,
    WorkflowListResponse,
    WorkflowExecutionCreate,
    WorkflowExecutionPublic,
)
from .services import (
    create_workflow,
    get_workflow,
    list_workflows,
    update_workflow,
    delete_workflow,
    create_execution,
    update_execution,
    list_executions,
)
from .executor import WorkflowExecutor, WorkflowExecutionError

logger = get_logger("api.workflows.router")

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("", response_model=WorkflowPublic, status_code=201)
async def create_workflow_endpoint(
    request: Request,
    payload: WorkflowCreate,
) -> WorkflowPublic:
    """
    Create a new workflow.
    
    Requires authentication. User ID is extracted from request state.
    """
    user_id = request.state.user.user_id
    workflow = await create_workflow(user_id, payload)
    return WorkflowPublic.model_validate(workflow, from_attributes=True)


@router.get("", response_model=WorkflowListResponse)
async def list_workflows_endpoint(request: Request) -> WorkflowListResponse:
    """
    List all workflows for the authenticated user.
    """
    user_id = request.state.user.user_id
    workflows = await list_workflows(user_id)
    return WorkflowListResponse(
        workflows=[
            WorkflowPublic.model_validate(w, from_attributes=True)
            for w in workflows
        ]
    )


@router.get("/{workflow_id}", response_model=WorkflowPublic)
async def get_workflow_endpoint(
    request: Request,
    workflow_id: int,
) -> WorkflowPublic:
    """
    Get a workflow by ID.
    
    Returns 404 if not found, 403 if unauthorized.
    """
    user_id = request.state.user.user_id
    workflow = await get_workflow(workflow_id, user_id)
    return WorkflowPublic.model_validate(workflow, from_attributes=True)


@router.put("/{workflow_id}", response_model=WorkflowPublic)
async def update_workflow_endpoint(
    request: Request,
    workflow_id: int,
    payload: WorkflowUpdate,
) -> WorkflowPublic:
    """
    Update a workflow.
    
    Returns 404 if not found, 403 if unauthorized.
    """
    user_id = request.state.user.user_id
    workflow = await update_workflow(workflow_id, user_id, payload)
    return WorkflowPublic.model_validate(workflow, from_attributes=True)


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow_endpoint(
    request: Request,
    workflow_id: int,
) -> None:
    """
    Delete a workflow (soft delete).
    
    Returns 404 if not found, 403 if unauthorized.
    """
    user_id = request.state.user.user_id
    await delete_workflow(workflow_id, user_id)


@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionPublic)
async def execute_workflow_endpoint(
    request: Request,
    workflow_id: int,
    payload: WorkflowExecutionCreate,
) -> WorkflowExecutionPublic:
    """
    Execute a workflow.
    
    Executes the workflow with the provided input data and returns results.
    For streaming execution, use the stream endpoint.
    """
    user_id = request.state.user.user_id
    
    # Get workflow
    workflow = await get_workflow(workflow_id, user_id)
    
    # Create execution record
    execution = await create_execution(
        workflow_id=workflow_id,
        user_id=user_id,
        input_data=payload.input_data,
    )
    
    try:
        # Execute workflow
        executor = WorkflowExecutor(user_id=user_id)
        result = await executor.execute(
            workflow=workflow,
            input_data=payload.input_data,
            execution=execution,
        )
        
        # Update execution with results
        status = 'completed' if result.get('success') else 'failed'
        output_data = result.get('output')
        error_message = result.get('error')
        
        execution = await update_execution(
            execution_id=execution.id,
            user_id=user_id,
            status=status,
            output_data=output_data,
            error_message=error_message,
        )
        
        return WorkflowExecutionPublic.model_validate(execution, from_attributes=True)
        
    except WorkflowExecutionError as e:
        # Update execution with error
        execution = await update_execution(
            execution_id=execution.id,
            user_id=user_id,
            status='failed',
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        execution = await update_execution(
            execution_id=execution.id,
            user_id=user_id,
            status='failed',
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail="Workflow execution failed")


@router.post("/{workflow_id}/execute/stream")
async def execute_workflow_stream_endpoint(
    request: Request,
    workflow_id: int,
    payload: WorkflowExecutionCreate,
) -> StreamingResponse:
    """
    Execute a workflow with streaming events.
    
    Streams execution events as Server-Sent Events (SSE).
    """
    import json
    from datetime import datetime
    
    user_id = request.state.user.user_id
    
    async def generate_events():
        try:
            # Get workflow
            workflow = await get_workflow(workflow_id, user_id)
            
            # Create execution record
            execution = await create_execution(
                workflow_id=workflow_id,
                user_id=user_id,
                input_data=payload.input_data,
            )
            
            yield f"event: execution_started\ndata: {json.dumps({'execution_id': execution.id})}\n\n"
            
            # Execute workflow
            executor = WorkflowExecutor(user_id=user_id)
            
            # TODO: Implement streaming execution with per-block events
            # For now, execute and stream final result
            result = await executor.execute(
                workflow=workflow,
                input_data=payload.input_data,
                execution=execution,
            )
            
            # Stream block execution events (simplified)
            yield f"event: block_executed\ndata: {json.dumps({'status': 'processing'})}\n\n"
            
            # Update execution
            status = 'completed' if result.get('success') else 'failed'
            execution = await update_execution(
                execution_id=execution.id,
                user_id=user_id,
                status=status,
                output_data=result.get('output'),
                error_message=result.get('error'),
            )
            
            yield f"event: execution_completed\ndata: {json.dumps({'execution_id': execution.id, 'status': status})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming execution failed: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/{workflow_id}/executions", response_model=list[WorkflowExecutionPublic])
async def list_executions_endpoint(
    request: Request,
    workflow_id: int,
    limit: int = Query(default=50, ge=1, le=100),
) -> list[WorkflowExecutionPublic]:
    """
    List executions for a workflow.
    """
    user_id = request.state.user.user_id
    executions = await list_executions(workflow_id, user_id, limit=limit)
    return [
        WorkflowExecutionPublic.model_validate(e, from_attributes=True)
        for e in executions
    ]


__all__ = ["router"]

