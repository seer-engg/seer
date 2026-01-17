"""
Workflow API router for CRUD and execution endpoints.
"""
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request
from langchain_core.messages import HumanMessage

from seer.agents.workflow_agent import (
    _current_thread_id,
    clear_proposed_spec_for_thread,
    clear_user_for_thread,
    create_workflow_chat_agent,
    extract_thinking_from_messages,
    get_proposed_spec_for_thread,
    set_user_for_thread,
    set_workflow_state_for_thread,
)
from seer.api.agents.checkpointer import _recreate_checkpointer, get_checkpointer
from seer.api.core.errors import AUTH_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.analytics import analytics
from seer.config import config
from seer.database import User, UserPublic
from seer.logger import get_logger
from seer.observability import (
    increment_chat_message_count,
)

from .chat_schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSession,
    ChatSessionCreate,
    ChatSessionWithMessages,
    WorkflowProposalActionResponse,
)
from .chat_services import (
    ChatOrchestrator,
    CheckpointerHealthService,
    IncompleteToolCallDetector,
    IncompleteToolCallRecoveryService,
    InterruptHandler,
    SessionService,
)
from .models import WorkflowProposalPublic
from .services import (
    accept_workflow_proposal,
    create_chat_session,
    create_workflow_proposal,
    get_chat_session,
    get_chat_session_by_thread_id,
    get_workflow,
    get_workflow_proposal,
    list_chat_sessions,
    load_chat_history,
    reject_workflow_proposal,
    save_chat_message,
    workflow_state_snapshot,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/workflow-agent", tags=["workflow-agent"])


def _require_user(request: Request) -> User:
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise_problem(
            type_uri=AUTH_PROBLEM,
            title="Unauthorized",
            detail="Unauthorized",
            status=401
        )
    return user


def _summarize_spec(spec: Dict[str, Any]) -> str:
    """Produce a short human summary for a WorkflowSpec."""
    if not spec:
        return "Workflow proposal"
    nodes = spec.get("nodes") or []
    node_types = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type", "node")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    if not node_types:
        return f"{len(nodes)} nodes"
    parts = [f"{count} {node_type}" for node_type, count in node_types.items()]
    return ", ".join(parts)


def _prepare_workflow_state(workflow, request_workflow_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare workflow state by merging saved and provided states."""
    workflow_state = deepcopy(workflow_state_snapshot(workflow))

    if request_workflow_state:
        workflow_state["nodes"] = request_workflow_state.get("nodes") or workflow_state.get("nodes", [])
        workflow_state["edges"] = request_workflow_state.get("edges") or workflow_state.get("edges", [])
        for key, value in request_workflow_state.items():
            if key not in ["nodes", "edges"]:
                workflow_state[key] = value

    workflow_state.setdefault("nodes", [])
    workflow_state.setdefault("edges", [])
    return workflow_state


def _extract_response_text(result: Dict[str, Any]) -> str:
    """Extract response text from agent result."""
    agent_messages = result.get("messages", []) if isinstance(result, dict) else []
    if agent_messages:
        last_msg = agent_messages[-1]
        return last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return "I'm here to help with your workflow!"


async def _verify_checkpoint_saved(checkpointer, thread_id: str) -> None:
    """Verify checkpoint was saved after agent invocation."""
    try:
        state_tuple = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
        if state_tuple:
            checkpoint_id = state_tuple.config.get("configurable", {}).get("checkpoint_id")
            logger.info("Checkpoint verified for thread %s, checkpoint_id=%s", thread_id, checkpoint_id)
        else:
            logger.warning("No checkpoint found for thread %s after agent invocation", thread_id)
    except Exception as e:
        logger.error("Error verifying checkpoint for thread %s: %s", thread_id, e, exc_info=True)


async def _maybe_create_proposal_from_spec(
    workflow,
    session,
    user,
    model_name: str,
    proposal_payload: Optional[Dict[str, Any]],
) -> Tuple[Optional[Any], Optional[WorkflowProposalPublic], Optional[str]]:
    """
    Persist workflow proposal if the agent provided a spec payload.

    Returns:
        Tuple of (proposal, proposal_public, error_message)
    """
    if not proposal_payload:
        return None, None, None

    spec = proposal_payload.get("spec")
    if not isinstance(spec, dict):
        return None, None, "Workflow spec payload is missing or malformed."

    summary = proposal_payload.get("summary") or _summarize_spec(spec)
    try:
        proposal = await create_workflow_proposal(
            workflow=workflow,
            session=session,
            user=user,
            summary=summary,
            spec=spec,
            metadata={"model": model_name},
        )
    except HTTPException as exc:
        error_detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return None, None, error_detail
    await proposal.fetch_related('created_by', 'workflow', 'session')
    proposal_public = WorkflowProposalPublic.model_validate(proposal, from_attributes=True)

    # Capture workflow proposal creation event
    analytics.capture(
        distinct_id=user.user_id,
        event="workflow_proposal_created",
        properties={
            "proposal_id": proposal.id,
            "workflow_id": workflow.workflow_id if workflow else None,
            "session_id": session.id if session else None,
            "model": model_name,
            "spec_node_count": len(spec.get("nodes", [])),
            "deployment_mode": config.seer_mode,
        },
    )

    return proposal, proposal_public, None


@router.post("/{workflow_id}/chat", response_model=ChatResponse)
async def chat_with_workflow_endpoint(
    request: Request,
    workflow_id: str,
    chat_request: ChatRequest,
) -> ChatResponse:
    """
    Chat with AI assistant about workflow.

    The assistant can analyze the workflow and suggest edits.
    Supports session persistence and human-in-the-loop interrupts.
    """
    logger.info("Chat request received: workflow_id=%s, message_length=%d", workflow_id, len(chat_request.message))
    user = _require_user(request)
    workflow = await get_workflow(user, workflow_id)

    # Chat limit check moved to UsageLimitMiddleware
    model = chat_request.model or config.default_llm_model
    checkpointer = await get_checkpointer()

    # Get or create session
    session, thread_id, session_id = await SessionService.get_or_create_session(
        workflow=workflow,
        user=user,
        thread_id=chat_request.thread_id,
        session_id=chat_request.session_id,
    )

    # Prepare workflow state
    workflow_state = _prepare_workflow_state(workflow, chat_request.workflow_state)
    set_workflow_state_for_thread(thread_id, workflow_state)
    set_user_for_thread(thread_id, user)

    # Create agent
    agent = create_workflow_chat_agent(
        model=model,
        checkpointer=checkpointer,
        workflow_state=workflow_state,
    )

    user_msg = HumanMessage(content=chat_request.message)

    # Save user message
    await save_chat_message(
        session_id=session_id,
        role="user",
        content=chat_request.message,
    )

    # Track user message (global count, not per-workflow)
    await increment_chat_message_count(user)

    analytics.capture(
        distinct_id=user.user_id,
        event="chat_agent_message",
        properties={
            "workflow_id": workflow_id,
            "session_id": session_id,
            "message_role": "user",
            "message_length": len(chat_request.message),
            "deployment_mode": config.seer_mode,
        },
    )

    try:
        config_dict = {"configurable": {"thread_id": thread_id}}

        # Initialize orchestrator
        orchestrator = ChatOrchestrator(
            agent=agent,
            checkpointer=checkpointer,
            health_service=CheckpointerHealthService(),
            detector=IncompleteToolCallDetector(),
            recovery_service=IncompleteToolCallRecoveryService(),
            reconnect_func=_recreate_checkpointer,
        )

        # Invoke agent with health checks
        result = await orchestrator.invoke_with_health_checks(user_msg, config_dict)

        # Detect interrupts
        interrupt_required, interrupt_data = InterruptHandler.extract_interrupt_from_result(result)
        if not interrupt_required:
            state_interrupt_required, state_interrupt_data = await InterruptHandler.extract_interrupt_from_state(
                agent, config_dict
            )
            if state_interrupt_required:
                interrupt_required = True
                interrupt_data = state_interrupt_data

        # Extract response and messages
        agent_messages = result.get("messages", []) if isinstance(result, dict) else []
        response_text = _extract_response_text(result)
        logger.info("Agent completed for thread %s, response_length=%d, interrupt_required=%s", thread_id, len(response_text), interrupt_required)

        # Verify checkpoint
        if checkpointer and thread_id:
            await _verify_checkpoint_saved(checkpointer, thread_id)

        # Extract thinking and proposal
        thinking_steps = extract_thinking_from_messages(agent_messages)
        proposal_payload = get_proposed_spec_for_thread(thread_id)
        proposal, proposal_public, proposal_error = await _maybe_create_proposal_from_spec(
            workflow=workflow,
            session=session,
            user=user,
            model_name=model,
            proposal_payload=proposal_payload,
        )

        # Save assistant message
        await save_chat_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            thinking="\n".join(thinking_steps) if thinking_steps else None,
            suggested_edits=proposal_payload,
            proposal=proposal,
        )

        # Track assistant message (global count, not per-workflow)
        await increment_chat_message_count(user)

        analytics.capture(
            distinct_id=user.user_id,
            event="chat_agent_message",
            properties={
                "workflow_id": workflow_id,
                "session_id": session_id,
                "message_role": "assistant",
                "message_length": len(response_text),
                "model": model,
                "created_proposal": proposal_public is not None,
                "deployment_mode": config.seer_mode,
            },
        )

        analytics.flush()

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
        logger.error("Error in workflow chat: %s", e, exc_info=True)
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Chat processing failed",
            detail=f"Failed to process chat request: {str(e)}",
            status=500
        )
    finally:
        clear_proposed_spec_for_thread(thread_id)
        clear_user_for_thread(thread_id)


@router.post("/{workflow_id}/chat/sessions", response_model=ChatSession)
async def create_chat_session_endpoint(
    request: Request,
    workflow_id: str,
    session_data: ChatSessionCreate,
) -> ChatSession:
    """Create a new chat session."""
    print(f"Creating chat session for workflow {workflow_id}")
    user = _require_user(request)
    workflow = await get_workflow(user, workflow_id)

    thread_id = f"workflow-{workflow_id}-{uuid.uuid4().hex}"
    session = await create_chat_session(
        workflow=workflow,
        user=user,
        thread_id=thread_id,
        title=session_data.title,
    )

    return ChatSession(
        id=session.id,
        workflow_id=workflow.workflow_id,
        user=UserPublic.model_validate(session.user, from_attributes=True),
        thread_id=session.thread_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/{workflow_id}/chat/sessions", response_model=list[ChatSession])
async def list_chat_sessions_endpoint(
    request: Request,
    workflow_id: str,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[ChatSession]:
    """List chat sessions for a workflow."""
    print(f"Listing chat sessions for workflow {workflow_id}")
    user = _require_user(request)
    workflow = await get_workflow(user, workflow_id)
    sessions = await list_chat_sessions(workflow, user, limit=limit, offset=offset)

    return [
        ChatSession(
            id=session.id,
            workflow_id=workflow.workflow_id,
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
    workflow_id: str,
    session_id: int,
) -> ChatSessionWithMessages:
    """Get a chat session with its messages."""
    user = _require_user(request)
    workflow = await get_workflow(user, workflow_id)
    session = await get_chat_session(session_id, workflow)

    messages = await load_chat_history(session_id)

    return ChatSessionWithMessages(
        id=session.id,
        workflow_id=workflow.workflow_id,
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
    workflow_id: str,
    resume_data: Dict[str, Any],
) -> ChatResponse:
    """
    Resume a chat session after an interrupt (e.g., clarification question).

    This endpoint handles resuming agent execution after a LangGraph interrupt.
    The resume_data should contain a Command object with resume information.
    """
    from langgraph.types import Command

    logger.info("Resume request received: workflow_id=%s", workflow_id)
    user = _require_user(request)

    # Verify workflow exists
    workflow = await get_workflow(user, workflow_id)

    # Extract thread_id and command from resume_data
    thread_id = resume_data.get("thread_id")
    if not thread_id:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing thread_id",
            detail="thread_id is required in resume_data",
            status=400
        )

    command_data = resume_data.get("command", {})
    if not command_data:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing command",
            detail="command is required in resume_data",
            status=400
        )

    # Get checkpointer
    checkpointer = await get_checkpointer()

    # Get session by thread_id
    session = await get_chat_session_by_thread_id(thread_id, workflow)
    if not session:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Session not found",
            detail=f"Chat session not found for thread_id: {thread_id}",
            status=404
        )

    session_id = session.id

    # Get current workflow state (deep copy to avoid mutating DB graph)
    workflow_state = deepcopy(workflow_state_snapshot(workflow))

    # Create agent
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
        thinking_steps = extract_thinking_from_messages(agent_messages)

        proposal_payload = get_proposed_spec_for_thread(thread_id)
        proposal, proposal_public, proposal_error = await _maybe_create_proposal_from_spec(
            workflow=workflow,
            session=session,
            user=user,
            model_name=config.default_llm_model,
            proposal_payload=proposal_payload,
        )

        # Save assistant message to database
        await save_chat_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            thinking="\n".join(thinking_steps) if thinking_steps else None,
            suggested_edits=proposal_payload,
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
        logger.error("Error resuming chat: %s", e, exc_info=True)
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Chat resume failed",
            detail=f"Failed to resume chat: {str(e)}",
            status=500
        )
    finally:
        # Reset context variable
        if token is not None:
            _current_thread_id.reset(token)
        clear_proposed_spec_for_thread(thread_id)


@router.get("/{workflow_id}/proposals/{proposal_id}", response_model=WorkflowProposalPublic)
async def get_proposal_endpoint(
    request: Request,
    workflow_id: str,
    proposal_id: int,
) -> WorkflowProposalPublic:
    """Fetch a single workflow proposal."""
    workflow = await get_workflow(_require_user(request), workflow_id)
    proposal = await get_workflow_proposal(workflow, proposal_id)
    await proposal.fetch_related('created_by', 'workflow', 'session')
    return WorkflowProposalPublic.model_validate(proposal, from_attributes=True)


@router.post("/{workflow_id}/proposals/{proposal_id}/accept", response_model=WorkflowProposalActionResponse)
async def accept_proposal_endpoint(
    request: Request,
    workflow_id: str,
    proposal_id: int,
) -> WorkflowProposalActionResponse:
    """Accept a workflow proposal and apply its changes."""
    user = _require_user(request)
    workflow = await get_workflow(user, workflow_id)
    proposal, workflow = await accept_workflow_proposal(
        workflow,
        proposal_id,
        actor=user,
    )
    await proposal.fetch_related('created_by', 'workflow', 'session')

    # Capture proposal acceptance event
    analytics.capture(
        distinct_id=user.user_id,
        event="workflow_proposal_accepted",
        properties={
            "proposal_id": proposal_id,
            "workflow_id": workflow_id,
            "session_id": proposal.session.id if proposal.session else None,
            "deployment_mode": config.seer_mode,
        },
    )

    return WorkflowProposalActionResponse(
        proposal=WorkflowProposalPublic.model_validate(proposal, from_attributes=True),
        workflow_graph=workflow_state_snapshot(workflow),
    )


@router.post("/{workflow_id}/proposals/{proposal_id}/reject", response_model=WorkflowProposalActionResponse)
async def reject_proposal_endpoint(
    request: Request,
    workflow_id: str,
    proposal_id: int,
) -> WorkflowProposalActionResponse:
    """Reject a workflow proposal without applying changes."""
    user = _require_user(request)
    workflow = await get_workflow(user, workflow_id)
    proposal = await reject_workflow_proposal(workflow, proposal_id)
    await proposal.fetch_related('created_by', 'workflow', 'session')

    # Capture proposal rejection event
    analytics.capture(
        distinct_id=user.user_id,
        event="workflow_proposal_rejected",
        properties={
            "proposal_id": proposal_id,
            "workflow_id": workflow_id,
            "session_id": proposal.session.id if proposal.session else None,
            "deployment_mode": config.seer_mode,
        },
    )

    return WorkflowProposalActionResponse(
        proposal=WorkflowProposalPublic.model_validate(proposal, from_attributes=True),
        workflow_graph=None,
    )


__all__ = ["router"]
