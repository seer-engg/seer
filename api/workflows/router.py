"""
Workflow API router for CRUD and execution endpoints.
"""
from typing import Optional, Dict, List, Any, Tuple
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
from shared.logger import get_logger
from shared.config import config

from .models import (
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowPublic,
    WorkflowListResponse,
    WorkflowExecutionCreate,
    WorkflowExecutionPublic,
    WorkflowProposalPublic,
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
    get_execution,
    create_chat_session,
    get_chat_session,
    get_chat_session_by_thread_id,
    list_chat_sessions,
    save_chat_message,
    load_chat_history,
    update_chat_session_title,
    create_workflow_proposal,
    get_workflow_proposal,
    accept_workflow_proposal,
    reject_workflow_proposal,
    preview_patch_ops,
)
from .graph_builder import get_workflow_graph_builder
from .chat_schema import (
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSession,
    ChatSessionWithMessages,
    ChatMessage,
    InterruptResponse,
    WorkflowProposalActionResponse,
)
from .chat_agent import create_workflow_chat_agent, extract_thinking_from_messages, _current_thread_id
from .alias_utils import (
    build_template_reference_examples,
    collect_input_variables,
    derive_block_aliases,
)
from api.agents.checkpointer import get_checkpointer, get_checkpointer_with_retry, _recreate_checkpointer
import uuid
import json
import asyncio
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Import psycopg for error type checking
try:
    import psycopg
except ImportError:
    psycopg = None
from shared.database.models import User, UserPublic

logger = get_logger("api.workflows.router")

router = APIRouter(prefix="/workflows", tags=["workflows"])


def _summarize_patch_ops(patch_ops: List[Dict[str, Any]]) -> str:
    """Produce a short human summary for proposal patches."""
    descriptions = [
        op.get("description")
        for op in patch_ops
        if isinstance(op, dict) and op.get("description")
    ]
    if descriptions:
        return "; ".join(descriptions)
    return f"{len(patch_ops)} workflow change(s)"


def _extract_patch_ops_from_messages(agent_messages: List[Any]) -> List[Dict[str, Any]]:
    """Find workflow patch operations emitted by tools."""
    patch_ops: List[Dict[str, Any]] = []
    tool_call_to_result: Dict[str, ToolMessage] = {}
    
    for msg in agent_messages:
        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id:
                tool_call_to_result[tool_call_id] = msg
    
    recognized_tool_prefixes = (
        "add_workflow",
        "modify_workflow",
        "remove_workflow",
        "add_workflow_edge",
        "remove_workflow_edge",
    )
    
    for msg in agent_messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict):
                    tool_call_id = tool_call.get("id")
                    tool_name = tool_call.get("name", "")
                else:
                    tool_call_id = getattr(tool_call, "id", None)
                    tool_name = getattr(tool_call, "name", "")
                
                if not tool_name.startswith(recognized_tool_prefixes):
                    continue
                
                tool_result_msg = tool_call_to_result.get(tool_call_id)
                if not tool_result_msg:
                    continue
                
                try:
                    content = tool_result_msg.content if hasattr(tool_result_msg, "content") else str(tool_result_msg)
                    result_data = json.loads(content)
                    
                    if "error" in result_data:
                        continue
                    
                    if "op" in result_data:
                        patch_ops.append(result_data)
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    logger.warning(f"Failed to parse tool result for {tool_name}: {exc}")
    
    return patch_ops


async def _maybe_create_proposal_from_ops(
    workflow,
    session,
    user,
    model_name: str,
    patch_ops: List[Dict[str, Any]],
) -> Tuple[Optional[Any], Optional[WorkflowProposalPublic], Optional[str]]:
    """
    Persist workflow proposal if there are patch ops.
    
    Returns:
        Tuple of (proposal, proposal_public, error_message)
        If validation fails, returns (None, None, error_message)
    """
    if not patch_ops:
        return None, None, None
    
    preview_graph = None
    validation_error = None
    try:
        preview_graph = preview_patch_ops(getattr(workflow, "graph_data", {}), patch_ops)
    except (HTTPException, ValueError, Exception) as preview_error:
        # Extract error message from exception
        if isinstance(preview_error, HTTPException):
            error_detail = preview_error.detail
        elif hasattr(preview_error, 'args') and preview_error.args:
            error_detail = str(preview_error.args[0])
        else:
            error_detail = str(preview_error)
        
        # Extract more specific error message if it's a validation error
        if "tool_name is required" in error_detail:
            validation_error = "Invalid workflow proposal: Tool blocks require 'tool_name' in their configuration. Please ensure all tool blocks have a tool_name specified."
        elif "user_prompt is required" in error_detail:
            validation_error = "Invalid workflow proposal: LLM blocks require 'user_prompt' in their configuration."
        elif "condition is required" in error_detail:
            validation_error = "Invalid workflow proposal: If/else blocks require 'condition' in their configuration."
        elif "array_var" in error_detail or "item_var" in error_detail:
            validation_error = "Invalid workflow proposal: For loop blocks require 'array_var' and 'item_var' in their configuration."
        else:
            validation_error = f"Invalid workflow proposal: {error_detail}"
        
        logger.warning(f"Unable to build preview graph for proposal: {preview_error}")
    
    # Don't persist proposal if validation failed
    if validation_error:
        return None, None, validation_error
    
    summary = _summarize_patch_ops(patch_ops)
    proposal = await create_workflow_proposal(
        workflow=workflow,
        session=session,
        user=user,
        summary=summary,
        patch_ops=patch_ops,
        preview_graph=preview_graph,
        metadata={"model": model_name},
    )
    await proposal.fetch_related('created_by', 'workflow', 'session')
    proposal_public = WorkflowProposalPublic.model_validate(proposal, from_attributes=True)
    return proposal, proposal_public, None




@router.post("", response_model=WorkflowPublic, status_code=201)
async def create_workflow_endpoint(
    request: Request,
    payload: WorkflowCreate,
) -> WorkflowPublic:
    """
    Create a new workflow.
    
    Requires authentication in cloud mode. User ID is extracted from request state.
    """
    user = request.state.db_user
    workflow = await create_workflow(user, payload)
    return WorkflowPublic.model_validate(workflow, from_attributes=True)


@router.get("", response_model=WorkflowListResponse)
async def list_workflows_endpoint(request: Request) -> WorkflowListResponse:
    """
    List all workflows for the authenticated user (cloud mode) or all workflows (self-hosted mode).
    """
    user = request.state.db_user
    workflows = await list_workflows(user)
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
    
    Returns 404 if not found, 403 if unauthorized (cloud mode only).
    """
    workflow = await get_workflow(workflow_id)
    return WorkflowPublic.model_validate(workflow, from_attributes=True)


@router.put("/{workflow_id}", response_model=WorkflowPublic)
async def update_workflow_endpoint(
    request: Request,
    workflow_id: int,
    payload: WorkflowUpdate,
) -> WorkflowPublic:
    """
    Update a workflow.
    
    Returns 404 if not found, 403 if unauthorized (cloud mode only).
    """
    workflow = await update_workflow(workflow_id, payload)
    return WorkflowPublic.model_validate(workflow, from_attributes=True)


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow_endpoint(
    request: Request,
    workflow_id: int,
) -> None:
    """
    Delete a workflow (soft delete).
    
    Returns 404 if not found, 403 if unauthorized (cloud mode only).
    """
    await delete_workflow(workflow_id)


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
    user:User = request.state.db_user
    # Get workflow
    workflow = await get_workflow(workflow_id)
    
    # Create execution record
    execution = await create_execution(
        workflow_id=workflow_id,
        user=user,
        input_data=payload.input_data,
    )
    
    try:
        # Build and compile graph
        builder = await get_workflow_graph_builder()
        compiled_graph = await builder.get_compiled_graph(
            workflow_id=workflow_id,
            workflow=workflow,
            user_id=user.user_id,
        )
        
        # Prepare initial state
        block_aliases = derive_block_aliases(workflow.graph_data)
        config = {"configurable": {"thread_id": f"workflow_{execution.id}"}}
        initial_state = {
            "input_data": payload.input_data or {},
            "execution_id": execution.id,
            "user_id": user.user_id,
            "block_outputs": {},
            "loop_state": None,
            "block_aliases": block_aliases,
        }
        
        # Execute workflow
        result = await compiled_graph.ainvoke(initial_state, config)
        
        # Extract outputs from output blocks
        output_data = result.get("block_outputs", {})
        
        # Update execution with results
        execution = await update_execution(
            execution_id=execution.id,
            status='completed',
            output_data=output_data,
            error_message=None,
        )
        
        return WorkflowExecutionPublic.model_validate(execution, from_attributes=True)
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        execution = await update_execution(
            execution_id=execution.id,
            status='failed',
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


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
    
    user:User = request.state.db_user
    
    async def generate_events():
        try:
            # Get workflow
            workflow = await get_workflow(workflow_id)
            
            # Create execution record
            execution = await create_execution(
                workflow_id=workflow_id,
                user=user,
                input_data=payload.input_data,
            )
            
            yield f"event: execution_started\ndata: {json.dumps({'execution_id': execution.id})}\n\n"
            
            # Build and compile graph
            builder = await get_workflow_graph_builder()
            compiled_graph = await builder.get_compiled_graph(
                workflow_id=workflow_id,
                workflow=workflow,
                user_id=user.user_id,
            )
            
            # Prepare initial state
            block_aliases = derive_block_aliases(workflow.graph_data)
            config = {"configurable": {"thread_id": f"workflow_{execution.id}"}}
            initial_state = {
                "input_data": payload.input_data or {},
                "execution_id": execution.id,
                "user_id": user.user_id,
                "block_outputs": {},
                "loop_state": None,
                "block_aliases": block_aliases,
            }
            
            # Stream execution events
            output_data = {}
            async for event in compiled_graph.astream(initial_state, config):
                # Stream block execution events
                for node_name, node_output in event.items():
                    yield f"event: block_executed\ndata: {json.dumps({'node': node_name, 'status': 'completed'})}\n\n"
                    if isinstance(node_output, dict) and "block_outputs" in node_output:
                        output_data.update(node_output.get("block_outputs", {}))
            
            # Update execution
            execution = await update_execution(
                execution_id=execution.id,
                status='completed',
                output_data=output_data,
                error_message=None,
            )
            
            yield f"event: execution_completed\ndata: {json.dumps({'execution_id': execution.id, 'status': 'completed'})}\n\n"
            
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
    executions = await list_executions(workflow_id, limit=limit)
    return [
        WorkflowExecutionPublic.model_validate(e, from_attributes=True)
        for e in executions
    ]


@router.get("/{workflow_id}/executions/{execution_id}", response_model=WorkflowExecutionPublic)
async def get_execution_endpoint(
    request: Request,
    workflow_id: int,
    execution_id: int,
) -> WorkflowExecutionPublic:
    """
    Get a single execution by ID.
    """
    execution = await get_execution(execution_id, workflow_id)
    return WorkflowExecutionPublic.model_validate(execution, from_attributes=True)


# Include PR sync workflow
from .pr_sync import router as pr_sync_router
router.include_router(pr_sync_router)

# Chat endpoints

@router.post("/{workflow_id}/chat", response_model=ChatResponse)
async def chat_with_workflow_endpoint(
    request: Request,
    workflow_id: int,
    chat_request: ChatRequest,
) -> ChatResponse:
    """
    Chat with AI assistant about workflow.
    
    The assistant can analyze the workflow and suggest edits.
    Supports session persistence and human-in-the-loop interrupts.
    """
    logger.info(f"Chat request received: workflow_id={workflow_id}, message_length={len(chat_request.message)}")
    user:User = request.state.db_user
    
    # Verify workflow exists and user has access
    workflow = await get_workflow(workflow_id)
    
    # Get model from request or use default
    model = chat_request.model or config.default_llm_model
    
    # Get checkpointer for persistence
    checkpointer = await get_checkpointer()
    
    # Create or get chat session
    thread_id = chat_request.thread_id
    session_id = chat_request.session_id
    session = None
    
    if thread_id:
        # Try to find existing session by thread_id
        session = await get_chat_session_by_thread_id(thread_id, workflow_id)
        if session:
            session_id = session.id
    elif session_id:
        # Get session by ID
        session = await get_chat_session(session_id, workflow_id)
        thread_id = session.thread_id
    else:
        # Create new session
        thread_id = f"workflow-{workflow_id}-{uuid.uuid4().hex}"
        session = await create_chat_session(
            workflow_id=workflow_id,
            user=user,
            thread_id=thread_id,
        )
        session_id = session.id
    
    if session is None:
        thread_id = thread_id or f"workflow-{workflow_id}-{uuid.uuid4().hex}"
        session = await create_chat_session(
            workflow_id=workflow_id,
            user=user,
            thread_id=thread_id,
        )
        session_id = session.id
    
    # Get current workflow state
    workflow_state = {
        "nodes": workflow.graph_data.get("nodes", []) if workflow.graph_data else [],
        "edges": workflow.graph_data.get("edges", []) if workflow.graph_data else [],
    }
    
    # Merge with provided workflow state (in case frontend has unsaved changes)
    if chat_request.workflow_state:
        # Ensure nodes and edges keys exist
        provided_nodes = chat_request.workflow_state.get("nodes", [])
        provided_edges = chat_request.workflow_state.get("edges", [])
        workflow_state["nodes"] = provided_nodes if provided_nodes else workflow_state.get("nodes", [])
        workflow_state["edges"] = provided_edges if provided_edges else workflow_state.get("edges", [])
        # Merge any other keys
        for key, value in chat_request.workflow_state.items():
            if key not in ["nodes", "edges"]:
                workflow_state[key] = value
    
    # Ensure workflow_state always has nodes and edges keys (even if empty)
    if "nodes" not in workflow_state:
        workflow_state["nodes"] = []
    if "edges" not in workflow_state:
        workflow_state["edges"] = []
    
    graph_snapshot = {
        "nodes": workflow_state.get("nodes", []),
        "edges": workflow_state.get("edges", []),
    }
    block_aliases = derive_block_aliases(graph_snapshot)
    workflow_state["block_aliases"] = block_aliases
    workflow_state["template_reference_examples"] = build_template_reference_examples(block_aliases)
    workflow_state["input_variables"] = sorted(collect_input_variables(graph_snapshot))
    
    # Store workflow_state in context for tools to access
    from .chat_agent import set_workflow_state_for_thread, _current_thread_id
    if thread_id:
        set_workflow_state_for_thread(thread_id, workflow_state)
    
    # Create agent with checkpointer and workflow_state
    agent = create_workflow_chat_agent(
        model=model,
        checkpointer=checkpointer,
        workflow_state=workflow_state,
    )
    
    # Prepare messages - only pass the new user message
    # When using a checkpointer, LangGraph automatically loads the full state from the checkpointer
    # We should NOT manually load messages as it conflicts with the checkpointer's state management
    user_msg = HumanMessage(content=chat_request.message)
    
    # Save user message to database for display purposes
    await save_chat_message(
        session_id=session_id,
        role="user",
        content=chat_request.message,
    )
    
    # Helper function to invoke agent with timeout
    async def invoke_agent_with_timeout(agent, messages, config, timeout=300.0):
        """Invoke agent with timeout to prevent indefinite hangs."""
        thread_id = config.get('configurable', {}).get('thread_id') if config else None
        # Set thread_id in context variable for tools to access
        if thread_id:
            token = _current_thread_id.set(thread_id)
        else:
            token = None
        try:
            return await asyncio.wait_for(
                agent.ainvoke(messages, config=config),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Agent invocation timed out after {timeout} seconds for thread {thread_id or 'unknown'}")
            raise HTTPException(
                status_code=504,
                detail="Request timed out. The agent took too long to respond."
            )
        finally:
            # Reset context variable
            if token is not None:
                _current_thread_id.reset(token)
    
    try:
        # Invoke agent with thread configuration
        # The checkpointer will automatically load the full conversation history
        config_dict = {
            "configurable": {
                "thread_id": thread_id,
            },
        }
        
        # Check for incomplete tool calls in state before invoking
        has_incomplete_tool_calls = False
        if checkpointer and thread_id:
            logger.debug(f"Checking checkpointer health for thread {thread_id}")
            try:
                # Check health and reconnect if needed
                checkpointer = await get_checkpointer_with_retry()
                if checkpointer is None:
                    logger.warning("Checkpointer unavailable, proceeding without state check")
                    has_incomplete_tool_calls = False
                else:
                    current_state = await agent.aget_state(config_dict)
                    messages = current_state.values.get("messages", [])
                    
                    # Check for incomplete tool calls
                    for i, msg in enumerate(messages):
                        # Check if this is an AIMessage with tool_calls
                        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                            # Extract tool_call IDs from the AIMessage
                            # Structure: msg.tool_calls = [{"id": "call_123", "name": "tool_name", "args": {...}, "type": "tool_call"}, ...]
                            tool_call_ids = set()
                            for tc in msg.tool_calls:
                                # Handle both dict and object formats
                                if isinstance(tc, dict):
                                    tool_call_id = tc.get("id")
                                else:
                                    tool_call_id = getattr(tc, "id", None)
                                if tool_call_id:
                                    tool_call_ids.add(tool_call_id)
                            
                            if not tool_call_ids:
                                continue
                                
                            # Check if following messages contain ToolMessages with matching tool_call_ids
                            # ToolMessage structure: ToolMessage(content="...", tool_call_id="call_123")
                            following_msgs = messages[i+1:i+1+len(tool_call_ids)*2]  # Allow some buffer
                            tool_response_ids = set()
                            
                            for m in following_msgs:
                                if isinstance(m, ToolMessage):
                                    # ToolMessage has tool_call_id attribute
                                    tool_call_id = getattr(m, "tool_call_id", None)
                                    if tool_call_id:
                                        tool_response_ids.add(tool_call_id)
                                elif isinstance(m, dict) and m.get("type") == "tool":
                                    # Handle dict format
                                    tool_call_id = m.get("tool_call_id")
                                    if tool_call_id:
                                        tool_response_ids.add(tool_call_id)
                            
                            # If any tool_call_ids don't have corresponding ToolMessages, it's incomplete
                            if tool_call_ids - tool_response_ids:
                                has_incomplete_tool_calls = True
                                logger.warning(
                                    f"Found incomplete tool calls. Missing responses for: {tool_call_ids - tool_response_ids}"
                                )
                                break
            except (Exception, ConnectionError, EOFError) as e:
                # Check if it's a connection error
                is_connection_error = (
                    (psycopg and isinstance(e, psycopg.OperationalError)) or
                    isinstance(e, ConnectionError) or
                    isinstance(e, EOFError) or
                    "connection is closed" in str(e).lower() or
                    "ssl syscall error" in str(e).lower()
                )
                
                if is_connection_error:
                    logger.warning(f"Connection error during state check: {e}, attempting reconnection...")
                    try:
                        checkpointer = await _recreate_checkpointer()
                        if checkpointer:
                            # Retry once after reconnection
                            try:
                                current_state = await agent.aget_state(config_dict)
                                messages = current_state.values.get("messages", [])
                                
                                # Check for incomplete tool calls (same logic as above)
                                for i, msg in enumerate(messages):
                                    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                                        tool_call_ids = set()
                                        for tc in msg.tool_calls:
                                            if isinstance(tc, dict):
                                                tool_call_id = tc.get("id")
                                            else:
                                                tool_call_id = getattr(tc, "id", None)
                                            if tool_call_id:
                                                tool_call_ids.add(tool_call_id)
                                        
                                        if not tool_call_ids:
                                            continue
                                        
                                        following_msgs = messages[i+1:i+1+len(tool_call_ids)*2]
                                        tool_response_ids = set()
                                        
                                        for m in following_msgs:
                                            if isinstance(m, ToolMessage):
                                                tool_call_id = getattr(m, "tool_call_id", None)
                                                if tool_call_id:
                                                    tool_response_ids.add(tool_call_id)
                                            elif isinstance(m, dict) and m.get("type") == "tool":
                                                tool_call_id = m.get("tool_call_id")
                                                if tool_call_id:
                                                    tool_response_ids.add(tool_call_id)
                                        
                                        if tool_call_ids - tool_response_ids:
                                            has_incomplete_tool_calls = True
                                            logger.warning(
                                                f"Found incomplete tool calls after reconnection. Missing responses for: {tool_call_ids - tool_response_ids}"
                                            )
                                            break
                            except Exception as retry_error:
                                logger.error(f"State check failed after reconnection: {retry_error}")
                                has_incomplete_tool_calls = False
                        else:
                            logger.warning("Failed to recreate checkpointer, proceeding without state check")
                            has_incomplete_tool_calls = False
                    except Exception as reconnect_error:
                        logger.error(f"Error during checkpointer reconnection: {reconnect_error}")
                        has_incomplete_tool_calls = False
                else:
                    logger.warning(f"Error checking state for incomplete tool calls: {e}. Proceeding with normal invocation.")
                    has_incomplete_tool_calls = False
        
        # Handle incomplete tool calls if detected
        if has_incomplete_tool_calls:
            logger.warning(f"Incomplete tool calls detected in thread {thread_id}, attempting recovery...")
            
            # Option A: Get previous checkpoint without incomplete calls
            if checkpointer:
                try:
                    # Ensure checkpointer is healthy before listing checkpoints
                    checkpointer = await get_checkpointer_with_retry()
                    if checkpointer is None:
                        logger.warning("Checkpointer unavailable for checkpoint recovery, deleting thread and starting fresh")
                        # Get a fresh checkpointer for deletion
                        fresh_checkpointer = await get_checkpointer()
                        if fresh_checkpointer:
                            if hasattr(fresh_checkpointer, 'adelete_thread'):
                                await fresh_checkpointer.adelete_thread(thread_id)
                            else:
                                await asyncio.to_thread(fresh_checkpointer.delete_thread, thread_id)
                        result = await invoke_agent_with_timeout(
                            agent,
                            {"messages": [user_msg]},
                            config_dict,
                        )
                    else:
                        # Add timeout to prevent hanging on slow database operations
                        checkpoints = []
                        try:
                            async def list_checkpoints():
                                return [c async for c in checkpointer.alist(config_dict)]
                            checkpoints = await asyncio.wait_for(
                                list_checkpoints(),
                                timeout=10.0  # 10 second timeout for listing checkpoints
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"Checkpoint listing timed out for thread {thread_id}")
                            checkpoints = []
                        # Find the last checkpoint that doesn't have incomplete tool calls
                        safe_checkpoint = None
                        for checkpoint_tuple in reversed(checkpoints[:-1]):  # Skip the latest (incomplete) one
                            checkpoint_messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
                            # Quick check if this checkpoint is safe
                            # Note: checkpoint messages are typically dicts, not message objects
                            has_incomplete = False
                            for j, chk_msg in enumerate(checkpoint_messages):
                                # Handle both dict and message object formats
                                msg_type = None
                                tool_calls = None
                                
                                if isinstance(chk_msg, AIMessage):
                                    msg_type = "ai"
                                    tool_calls = getattr(chk_msg, "tool_calls", None)
                                elif isinstance(chk_msg, dict):
                                    msg_type = chk_msg.get("type") or chk_msg.get("role", "")
                                    tool_calls = chk_msg.get("tool_calls")
                                
                                # Check if this is an AIMessage with tool_calls
                                if msg_type in ("ai", "assistant") and tool_calls:
                                    chk_tool_call_ids = set()
                                    for tc in tool_calls:
                                        if isinstance(tc, dict):
                                            tool_call_id = tc.get("id")
                                        else:
                                            tool_call_id = getattr(tc, "id", None)
                                        if tool_call_id:
                                            chk_tool_call_ids.add(tool_call_id)
                                    
                                    if chk_tool_call_ids:
                                        chk_following = checkpoint_messages[j+1:j+1+len(chk_tool_call_ids)*2]
                                        chk_response_ids = set()
                                        
                                        for m in chk_following:
                                            if isinstance(m, ToolMessage):
                                                tool_call_id = getattr(m, "tool_call_id", None)
                                                if tool_call_id:
                                                    chk_response_ids.add(tool_call_id)
                                            elif isinstance(m, dict):
                                                m_type = m.get("type") or m.get("role", "")
                                                if m_type == "tool":
                                                    tool_call_id = m.get("tool_call_id")
                                                    if tool_call_id:
                                                        chk_response_ids.add(tool_call_id)
                                        
                                        if chk_tool_call_ids - chk_response_ids:
                                            has_incomplete = True
                                            break
                            if not has_incomplete:
                                safe_checkpoint = checkpoint_tuple
                                break
                    
                    if safe_checkpoint:
                        prev_config = {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_id": safe_checkpoint.config["configurable"]["checkpoint_id"]
                            }
                        }
                        logger.info(f"Resuming from safe checkpoint: {prev_config['configurable']['checkpoint_id']}")
                        result = await invoke_agent_with_timeout(
                            agent,
                            {"messages": [user_msg]},
                            prev_config,
                        )
                    else:
                        # Option B: Delete thread and start fresh
                        logger.warning(f"No safe checkpoint found, deleting thread {thread_id} and starting fresh")
                        # Use async delete_thread if available, otherwise wrap sync call
                        if hasattr(checkpointer, 'adelete_thread'):
                            await checkpointer.adelete_thread(thread_id)
                        else:
                            await asyncio.to_thread(checkpointer.delete_thread, thread_id)
                        result = await invoke_agent_with_timeout(
                            agent,
                            {"messages": [user_msg]},
                            config_dict,
                        )
                except (Exception, ConnectionError, EOFError) as e:
                    # Check if it's a connection error
                    is_connection_error = (
                        (psycopg and isinstance(e, psycopg.OperationalError)) or
                        isinstance(e, ConnectionError) or
                        isinstance(e, EOFError) or
                        "connection is closed" in str(e).lower() or
                        "ssl syscall error" in str(e).lower()
                    )
                    
                    if is_connection_error:
                        logger.warning(f"Connection error during checkpoint recovery: {e}, attempting reconnection...")
                        try:
                            checkpointer = await _recreate_checkpointer()
                            if checkpointer:
                                # Try to delete thread with reconnected checkpointer
                                if hasattr(checkpointer, 'adelete_thread'):
                                    await checkpointer.adelete_thread(thread_id)
                                else:
                                    await asyncio.to_thread(checkpointer.delete_thread, thread_id)
                                result = await invoke_agent_with_timeout(
                                    agent,
                                    {"messages": [user_msg]},
                                    config_dict,
                                )
                            else:
                                logger.error("Failed to recreate checkpointer, proceeding without deletion")
                                result = await invoke_agent_with_timeout(
                                    agent,
                                    {"messages": [user_msg]},
                                    config_dict,
                                )
                        except Exception as reconnect_error:
                            logger.error(f"Error during checkpointer reconnection in recovery: {reconnect_error}")
                            result = await invoke_agent_with_timeout(
                                agent,
                                {"messages": [user_msg]},
                                config_dict,
                            )
                    else:
                        logger.error(f"Error recovering from incomplete state: {e}", exc_info=True)
                        # Fallback: delete thread
                        fresh_checkpointer = await get_checkpointer()
                        if fresh_checkpointer:
                            if hasattr(fresh_checkpointer, 'adelete_thread'):
                                await fresh_checkpointer.adelete_thread(thread_id)
                            else:
                                await asyncio.to_thread(fresh_checkpointer.delete_thread, thread_id)
                        result = await invoke_agent_with_timeout(
                            agent,
                            {"messages": [user_msg]},
                            config_dict,
                        )
            else:
                # No checkpointer, can't recover - this shouldn't happen but handle gracefully
                logger.error("No checkpointer available for state recovery")
                result = await invoke_agent_with_timeout(
                    agent,
                    {"messages": [user_msg]},
                    config_dict,
                )
        else:
            # Normal invocation - state is clean
            logger.info(f"Invoking agent for thread {thread_id} with checkpointer={'enabled' if checkpointer else 'disabled'}")
            result = await invoke_agent_with_timeout(
                agent,
                {"messages": [user_msg]},
                config_dict,
            )
            logger.debug(f"Agent invocation completed for thread {thread_id}, checkpoint should be saved automatically by LangGraph")
        
        # Check for interrupts (from ask_clarifying_question or other interrupt calls)
        interrupt_required = False
        interrupt_data = None
        
        # Check if result indicates an interrupt
        if isinstance(result, dict):
            # Check for interrupt in result
            if "__interrupt__" in result:
                interrupt_required = True
                interrupts = result["__interrupt__"]
                # Handle list of Interrupt objects
                if isinstance(interrupts, list) and len(interrupts) > 0:
                    first_interrupt = interrupts[0]
                    # Extract value from Interrupt object
                    if hasattr(first_interrupt, 'value'):
                        interrupt_data = first_interrupt.value if isinstance(first_interrupt.value, dict) else {"value": first_interrupt.value}
                    elif isinstance(first_interrupt, dict):
                        interrupt_data = first_interrupt.get('value', first_interrupt)
                    else:
                        interrupt_data = {"value": str(first_interrupt)}
                elif isinstance(interrupts, dict):
                    interrupt_data = interrupts
                else:
                    interrupt_data = {"value": str(interrupts)}
            # Also check state for interrupts
            elif "interrupt" in result:
                interrupt_required = True
                interrupt_data = result["interrupt"] if isinstance(result["interrupt"], dict) else {"value": result["interrupt"]}
        
        # Check current state for interrupts
        try:
            current_state = await agent.aget_state(config_dict)
            if hasattr(current_state, "interrupt") and current_state.interrupt:
                interrupt_required = True
                if isinstance(current_state.interrupt, list) and len(current_state.interrupt) > 0:
                    # Handle list of interrupts
                    first_interrupt = current_state.interrupt[0]
                    if hasattr(first_interrupt, 'value'):
                        interrupt_data = first_interrupt.value if isinstance(first_interrupt.value, dict) else {"value": first_interrupt.value}
                    elif isinstance(first_interrupt, dict):
                        interrupt_data = first_interrupt.get('value', first_interrupt)
                    else:
                        interrupt_data = {"value": str(first_interrupt)}
                elif isinstance(current_state.interrupt, dict):
                    interrupt_data = current_state.interrupt
                else:
                    interrupt_data = {"value": current_state.interrupt}
        except Exception as e:
            logger.debug(f"Could not check state for interrupts: {e}")
        
        # Extract response
        agent_messages = result.get("messages", []) if isinstance(result, dict) else []
        if not agent_messages:
            response_text = "I'm here to help with your workflow!"
        else:
            # Get last assistant message
            last_msg = agent_messages[-1]
            if hasattr(last_msg, "content"):
                response_text = last_msg.content
            else:
                response_text = str(last_msg)
        
        logger.info(f"Agent completed for thread {thread_id}, response_length={len(response_text)}, interrupt_required={interrupt_required}")
        
        # Verify checkpoint was saved after agent invocation
        if checkpointer and thread_id:
            try:
                # Verify checkpoint exists by getting the current state
                verify_config = {"configurable": {"thread_id": thread_id}}
                state_tuple = await checkpointer.aget_tuple(verify_config)
                if state_tuple:
                    checkpoint_id = state_tuple.config.get("configurable", {}).get("checkpoint_id")
                    logger.info(f"Checkpoint verified for thread {thread_id}, checkpoint_id={checkpoint_id}")
                else:
                    logger.warning(f"No checkpoint found for thread {thread_id} after agent invocation")
            except Exception as e:
                logger.error(f"Error verifying checkpoint for thread {thread_id}: {e}", exc_info=True)
        
        # Extract thinking steps
        thinking_steps = extract_thinking_from_messages(agent_messages)
        
        patch_ops = _extract_patch_ops_from_messages(agent_messages)
        proposal, proposal_public, proposal_error = await _maybe_create_proposal_from_ops(
            workflow=workflow,
            session=session,
            user=user,
            model_name=model,
            patch_ops=patch_ops,
        )
        
        # Save assistant message to database
        await save_chat_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            thinking="\n".join(thinking_steps) if thinking_steps else None,
            suggested_edits={"patch_ops": patch_ops} if patch_ops else None,
            proposal=proposal,
        )
        
        return ChatResponse(
            response=response_text,
            proposal=proposal_public,
            proposal_error=proposal_error,
            session_id=session_id,
            thread_id=thread_id,
            thinking=thinking_steps if thinking_steps else None,
            interrupt_required=interrupt_required,
            interrupt_data=interrupt_data,
        )
    except Exception as e:
        logger.error(f"Error in workflow chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )


@router.post("/{workflow_id}/chat/sessions", response_model=ChatSession)
async def create_chat_session_endpoint(
    request: Request,
    workflow_id: int,
    session_data: ChatSessionCreate,
) -> ChatSession:
    """Create a new chat session."""
    user:User = request.state.db_user
    workflow = await get_workflow(workflow_id)
    
    thread_id = f"workflow-{workflow_id}-{uuid.uuid4().hex}"
    session = await create_chat_session(
        workflow_id=workflow_id,
        user=user,
        thread_id=thread_id,
        title=session_data.title,
    )
    
    return ChatSession(
        id=session.id,
        workflow_id=workflow_id,  # Use the workflow_id parameter directly
        user=UserPublic.model_validate(session.user, from_attributes=True),
        thread_id=session.thread_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/{workflow_id}/chat/sessions", response_model=list[ChatSession])
async def list_chat_sessions_endpoint(
    request: Request,
    workflow_id: int,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[ChatSession]:
    """List chat sessions for a workflow."""
    user:User = request.state.db_user
    sessions = await list_chat_sessions(workflow_id, user, limit=limit, offset=offset)
    
    return [
        ChatSession(
            id=session.id,
            workflow_id=workflow_id,  # Use the workflow_id parameter directly
            user=UserPublic.model_validate(session.user, from_attributes=True),
            thread_id=session.thread_id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
        for session in sessions
    ]


@router.get("/{workflow_id}/chat/sessions/{session_id}", response_model=ChatSessionWithMessages)
async def get_chat_session_endpoint(
    request: Request,
    workflow_id: int,
    session_id: int,
) -> ChatSessionWithMessages:
    """Get a chat session with its messages."""
    user:User = request.state.db_user
    session = await get_chat_session(session_id, workflow_id)
    
    messages = await load_chat_history(session_id)
    
    return ChatSessionWithMessages(
        id=session.id,
        workflow_id=workflow_id,  # Use the workflow_id parameter directly
        user=UserPublic.model_validate(session.user, from_attributes=True),
        thread_id=session.thread_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[
            ChatMessage(
                id=msg.id,
                session_id=session_id,  # Use the session_id parameter directly
                role=msg.role,
                content=msg.content,
                thinking=msg.thinking,
                suggested_edits=msg.suggested_edits,
                proposal=WorkflowProposalPublic.model_validate(msg.proposal, from_attributes=True) if msg.proposal else None,
                metadata=msg.metadata,
                created_at=msg.created_at,
            )
            for msg in messages
        ],
    )


@router.post("/{workflow_id}/chat/resume")
async def resume_chat_endpoint(
    request: Request,
    workflow_id: int,
    resume_data: Dict[str, Any],
) -> ChatResponse:
    """
    Resume a chat session after an interrupt (e.g., clarification question).
    
    This endpoint handles resuming agent execution after a LangGraph interrupt.
    The resume_data should contain a Command object with resume information.
    """
    from langgraph.types import Command
    
    logger.info(f"Resume request received: workflow_id={workflow_id}")
    user:User = request.state.db_user
    
    # Verify workflow exists
    workflow = await get_workflow(workflow_id)
    
    # Extract thread_id and command from resume_data
    thread_id = resume_data.get("thread_id")
    if not thread_id:
        raise HTTPException(
            status_code=400,
            detail="thread_id is required in resume_data"
        )
    
    command_data = resume_data.get("command", {})
    if not command_data:
        raise HTTPException(
            status_code=400,
            detail="command is required in resume_data"
        )
    
    # Get checkpointer
    checkpointer = await get_checkpointer()
    
    # Get session by thread_id
    session = await get_chat_session_by_thread_id(thread_id, workflow_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Chat session not found for thread_id: {thread_id}"
        )
    
    session_id = session.id
    
    # Get current workflow state
    workflow_state = {
        "nodes": workflow.graph_data.get("nodes", []) if workflow.graph_data else [],
        "edges": workflow.graph_data.get("edges", []) if workflow.graph_data else [],
    }
    
    # Create agent
    from .chat_agent import create_workflow_chat_agent
    agent = create_workflow_chat_agent(
        model=config.default_llm_model,
        checkpointer=checkpointer,
        workflow_state=workflow_state,
    )
    
    # Create Command object for resuming
    resume_command = Command(**command_data)
    
    # Resume agent execution
    config_dict = {
        "configurable": {
            "thread_id": thread_id,
        },
    }
    
    # Set thread_id in context variable for tools to access
    token = None
    if thread_id:
        token = _current_thread_id.set(thread_id)
    try:
        # Resume the agent with the command
        result = await agent.ainvoke(resume_command, config=config_dict)
        
        # Extract response
        agent_messages = result.get("messages", [])
        if not agent_messages:
            response_text = "I've received your response. Let me continue..."
        else:
            # Get last assistant message
            last_msg = agent_messages[-1]
            if hasattr(last_msg, "content"):
                response_text = last_msg.content
            else:
                response_text = str(last_msg)
        
        # Extract thinking steps
        from .chat_agent import extract_thinking_from_messages
        thinking_steps = extract_thinking_from_messages(agent_messages)
        
        patch_ops = _extract_patch_ops_from_messages(agent_messages)
        proposal, proposal_public, proposal_error = await _maybe_create_proposal_from_ops(
            workflow=workflow,
            session=session,
            user=user,
            model_name=config.default_llm_model,
            patch_ops=patch_ops,
        )
        
        # Save assistant message to database
        await save_chat_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            thinking="\n".join(thinking_steps) if thinking_steps else None,
            suggested_edits={"patch_ops": patch_ops} if patch_ops else None,
            proposal=proposal,
        )
        
        return ChatResponse(
            response=response_text,
            proposal=proposal_public,
            proposal_error=proposal_error,
            session_id=session_id,
            thread_id=thread_id,
            thinking=thinking_steps if thinking_steps else None,
            interrupt_required=False,
        )
    except Exception as e:
        logger.error(f"Error resuming chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume chat: {str(e)}"
        )
    finally:
        # Reset context variable
        if token is not None:
            _current_thread_id.reset(token)


@router.get("/{workflow_id}/proposals/{proposal_id}", response_model=WorkflowProposalPublic)
async def get_proposal_endpoint(
    request: Request,
    workflow_id: int,
    proposal_id: int,
) -> WorkflowProposalPublic:
    """Fetch a single workflow proposal."""
    proposal = await get_workflow_proposal(workflow_id, proposal_id)
    await proposal.fetch_related('created_by', 'workflow', 'session')
    return WorkflowProposalPublic.model_validate(proposal, from_attributes=True)


@router.post("/{workflow_id}/proposals/{proposal_id}/accept", response_model=WorkflowProposalActionResponse)
async def accept_proposal_endpoint(
    request: Request,
    workflow_id: int,
    proposal_id: int,
) -> WorkflowProposalActionResponse:
    """Accept a workflow proposal and apply its changes."""
    proposal, workflow = await accept_workflow_proposal(workflow_id, proposal_id)
    await proposal.fetch_related('created_by', 'workflow', 'session')
    return WorkflowProposalActionResponse(
        proposal=WorkflowProposalPublic.model_validate(proposal, from_attributes=True),
        workflow_graph=workflow.graph_data,
    )


@router.post("/{workflow_id}/proposals/{proposal_id}/reject", response_model=WorkflowProposalActionResponse)
async def reject_proposal_endpoint(
    request: Request,
    workflow_id: int,
    proposal_id: int,
) -> WorkflowProposalActionResponse:
    """Reject a workflow proposal without applying changes."""
    proposal = await reject_workflow_proposal(workflow_id, proposal_id)
    await proposal.fetch_related('created_by', 'workflow', 'session')
    return WorkflowProposalActionResponse(
        proposal=WorkflowProposalPublic.model_validate(proposal, from_attributes=True),
        workflow_graph=None,
    )


__all__ = ["router"]

