"""
Workflow API router for CRUD and execution endpoints.
"""
from typing import Optional, Dict, Any, Tuple
from fastapi import APIRouter, Request, HTTPException, Query
from shared.logger import get_logger
from shared.config import config
from shared.analytics import analytics
from copy import deepcopy

# Import refactored helpers
from .chat_helpers import (
    _get_or_create_session,
    _prepare_workflow_state,
    _setup_thread_context,
    _invoke_with_checkpoint_recovery,
    _check_for_interrupts,
    _extract_response_text,
    _verify_checkpoint_saved,
    _save_assistant_message_to_db,
    _track_assistant_message_analytics,
    _build_chat_response,
)

from .models import (
    WorkflowProposalPublic,
)
from .services import (
    get_workflow,
    create_chat_session,
    get_chat_session,
    list_chat_sessions,
    save_chat_message,
    load_chat_history,
    create_workflow_proposal,
    get_workflow_proposal,
    accept_workflow_proposal,
    reject_workflow_proposal,
    workflow_state_snapshot,
)

from .chat_schema import (
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSession,
    ChatSessionWithMessages,
    ChatMessage,
    WorkflowProposalActionResponse,
)
from agents.workflow_agent import (
    create_workflow_chat_agent,
    extract_thinking_from_messages,
    _current_thread_id,
    get_proposed_spec_for_thread,
    clear_proposed_spec_for_thread,
    clear_user_for_thread,
)
from api.agents.checkpointer import get_checkpointer
import uuid
from langchain_core.messages import HumanMessage

# Import psycopg for error type checking
try:
    import psycopg
except ImportError:
    psycopg = None
from shared.database.models import User, UserPublic, WorkflowProposal

logger = get_logger(__name__)

router = APIRouter(prefix="/workflow-agent", tags=["workflow-agent"])


def _require_user(request: Request) -> User:
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _validate_resume_data(resume_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Validate resume_data contains required fields. Returns (thread_id, command_data)."""
    thread_id = resume_data.get("thread_id")
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required in resume_data")
    command_data = resume_data.get("command", {})
    if not command_data:
        raise HTTPException(status_code=400, detail="command is required in resume_data")
    return thread_id, command_data


async def _get_session_for_thread(thread_id: str, workflow) -> Any:
    """Get chat session by thread_id with error handling."""
    from .services import get_chat_session_by_thread_id
    session = await get_chat_session_by_thread_id(thread_id, workflow)
    if not session:
        raise HTTPException(status_code=404, detail=f"Chat session not found for thread_id: {thread_id}")
    return session


async def _reload_proposal_with_relationships(proposal_id: int):
    """Reload proposal with all relationships (created_by, workflow, session)."""
    from shared.database.base import async_session_maker
    from sqlmodel import select
    from sqlalchemy.orm import selectinload
    from shared.database.models import WorkflowProposal

    async with async_session_maker() as session_db:
        stmt = (
            select(WorkflowProposal)
            .where(WorkflowProposal.id == proposal_id)
            .options(
                selectinload(WorkflowProposal.created_by),
                selectinload(WorkflowProposal.workflow),
                selectinload(WorkflowProposal.session)
            )
        )
        result = await session_db.execute(stmt)
        return result.scalar_one()


def _track_proposal_event(user: User, event_name: str, proposal_id: int, workflow_id: str, proposal) -> None:
    """Track proposal analytics event (accept/reject)."""
    analytics.capture(
        distinct_id=user.user_id,
        event=event_name,
        properties={
            "proposal_id": proposal_id,
            "workflow_id": workflow_id,
            "session_id": proposal.session.id if proposal.session else None,
            "deployment_mode": config.seer_mode,
        },
    )


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

    # Reload proposal with relationships
    from shared.database.base import async_session_maker
    from sqlmodel import select
    from sqlalchemy.orm import selectinload
    async with async_session_maker() as session_db:
        stmt = (
            select(WorkflowProposal)
            .where(WorkflowProposal.id == proposal.id)
            .options(
                selectinload(WorkflowProposal.created_by),
                selectinload(WorkflowProposal.workflow),
                selectinload(WorkflowProposal.session)
            )
        )
        result = await session_db.execute(stmt)
        proposal = result.scalar_one()

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

# Chat endpoints

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
    logger.info(f"Chat request received: workflow_id={workflow_id}, message_length={len(chat_request.message)}")

    try:
        # 1. Authentication and validation
        user = _require_user(request)
        workflow = await get_workflow(user, workflow_id)
        model = chat_request.model or config.default_llm_model
        checkpointer = await get_checkpointer()

        # 2. Get or create chat session
        session, session_id, thread_id = await _get_or_create_session(
            thread_id=chat_request.thread_id,
            session_id=chat_request.session_id,
            workflow=workflow,
            user=user,
            workflow_id=workflow_id,
        )

        # 3. Prepare workflow state (merge DB state with frontend changes)
        workflow_state = _prepare_workflow_state(
            workflow_state_snapshot=workflow_state_snapshot(workflow),
            provided_state=chat_request.workflow_state,
        )

        # 4. Setup thread context for tools
        _setup_thread_context(thread_id, workflow_state, user)

        # 5. Create agent
        agent = create_workflow_chat_agent(
            model=model,
            checkpointer=checkpointer,
            workflow_state=workflow_state,
        )

        # 6. Save user message
        user_msg = HumanMessage(content=chat_request.message)
        await save_chat_message(
            session_id=session_id,
            role="user",
            content=chat_request.message,
        )

        # Track user message
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

        # 7. Invoke agent with automatic checkpoint recovery
        config_dict = {"configurable": {"thread_id": thread_id}}
        result = await _invoke_with_checkpoint_recovery(
            agent=agent,
            checkpointer=checkpointer,
            thread_id=thread_id,
            config_dict=config_dict,
            user_msg=user_msg,
        )

        # 8. Check for interrupts
        interrupt_required, interrupt_data = await _check_for_interrupts(result, agent, config_dict)

        # 9. Extract response
        response_text = _extract_response_text(result)
        logger.info(f"Agent completed for thread {thread_id}, response_length={len(response_text)}, interrupt_required={interrupt_required}")

        # 10. Verify checkpoint was saved
        await _verify_checkpoint_saved(checkpointer, thread_id)

        # 11. Extract thinking and handle proposal
        agent_messages = result.get("messages", []) if isinstance(result, dict) else []
        thinking_steps = extract_thinking_from_messages(agent_messages)
        proposal_payload = get_proposed_spec_for_thread(thread_id)

        proposal, proposal_public, proposal_error = await _maybe_create_proposal_from_spec(
            workflow=workflow,
            session=session,
            user=user,
            model_name=model,
            proposal_payload=proposal_payload,
        )

        # 12. Save assistant message
        await _save_assistant_message_to_db(
            session_id=session_id,
            response_text=response_text,
            thinking_steps=thinking_steps,
            proposal_payload=proposal_payload,
            proposal=proposal,
        )

        # 13. Track analytics
        _track_assistant_message_analytics(
            user=user,
            workflow_id=workflow_id,
            session_id=session_id,
            response_text=response_text,
            model=model,
            proposal_public=proposal_public,
        )

        # Ensure PostHog events are sent before response returns
        analytics.flush()

        # 14. Build and return response
        return _build_chat_response(
            response_text=response_text,
            proposal_public=proposal_public,
            proposal_error=proposal_error,
            session_id=session_id,
            thread_id=thread_id,
            thinking_steps=thinking_steps,
            interrupt_required=interrupt_required,
            interrupt_data=interrupt_data,
        )

    except Exception as e:
        logger.error(f"Error in workflow chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
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
    from .chat_helpers import _extract_response_text, _save_assistant_message_to_db, _build_chat_response

    logger.info(f"Resume request received: workflow_id={workflow_id}")
    user = _require_user(request)
    workflow = await get_workflow(user, workflow_id)

    thread_id, command_data = _validate_resume_data(resume_data)
    checkpointer = await get_checkpointer()
    session = await _get_session_for_thread(thread_id, workflow)
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
        result = await agent.ainvoke(resume_command, config=config_dict)

        response_text = _extract_response_text(result)
        if not response_text or response_text == "No response from agent":
            response_text = "I've received your response. Let me continue..."

        agent_messages = result.get("messages", [])
        thinking_steps = extract_thinking_from_messages(agent_messages)
        proposal_payload = get_proposed_spec_for_thread(thread_id)

        proposal, proposal_public, proposal_error = await _maybe_create_proposal_from_spec(
            workflow=workflow, session=session, user=user,
            model_name=config.default_llm_model, proposal_payload=proposal_payload
        )

        await _save_assistant_message_to_db(session_id, response_text, thinking_steps, proposal_payload, proposal)

        return _build_chat_response(response_text, proposal_public, proposal_error, session_id, thread_id, thinking_steps, False, None)
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
    proposal = await _reload_proposal_with_relationships(proposal.id)
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
    proposal, workflow = await accept_workflow_proposal(workflow, proposal_id, actor=user)
    proposal = await _reload_proposal_with_relationships(proposal.id)
    _track_proposal_event(user, "workflow_proposal_accepted", proposal_id, workflow_id, proposal)
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
    proposal = await _reload_proposal_with_relationships(proposal.id)
    _track_proposal_event(user, "workflow_proposal_rejected", proposal_id, workflow_id, proposal)
    return WorkflowProposalActionResponse(
        proposal=WorkflowProposalPublic.model_validate(proposal, from_attributes=True),
        workflow_graph=None,
    )


__all__ = ["router"]
