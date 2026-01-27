# pylint: disable=too-many-lines # Reason: Complex API router with multiple endpoints, requires architectural refactoring to split
"""
Workflow API router for CRUD and execution endpoints.
"""
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from tortoise.exceptions import DoesNotExist

from seer.agents.nexus import (
    _current_thread_id,
    clear_proposed_spec_for_thread,
    clear_user_for_thread,
    create_nexus_chat_agent,
    extract_thinking_from_messages,
    get_proposed_spec_for_thread,
    set_user_for_thread,
    set_workflow_state_for_thread,
)
from seer.agents.nexus.cost_callback import (
    CostCapCallbackHandler,
    set_chat_runtime_context,
    clear_chat_runtime_context,
)
from seer.api.agents.checkpointer import _recreate_checkpointer, get_checkpointer
from seer.api.core.errors import AUTH_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.config import config
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.database import User, UserPublic
from seer.database.models import UserSettings
from seer.logger import get_logger
from seer.observability import (
    increment_chat_message_count,
)
from seer.observability.exceptions import RunCostCapExceeded

from .chat_schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSession,
    ChatSessionCreate,
    ChatSessionWithMessages,
    DiscoveryChatRequest,
    WorkflowCreationMode,
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
    create_discovery_chat_session,
    create_workflow_proposal,
    get_chat_session,
    get_chat_session_by_thread_id,
    get_discovery_chat_session_by_thread_id,
    get_user_workflow_creation_mode,
    get_workflow,
    get_workflow_proposal,
    list_chat_sessions,
    load_chat_history,
    reject_workflow_proposal,
    save_chat_message,
    update_user_workflow_creation_mode,
    workflow_state_snapshot,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/nexus", tags=["nexus"])


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


async def _prepare_workflow_state(workflow, request_workflow_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare workflow state by merging saved and provided states."""
    workflow_state = deepcopy(await workflow_state_snapshot(workflow))

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
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Checkpoint verification is non-critical, log and continue
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

    return proposal, proposal_public, None


async def _get_or_create_discovery_session(
    user: User,
    thread_id: str | None,
    creation_mode: WorkflowCreationMode,
) -> Tuple[Any, str, str]:
    """
    Get or create discovery chat session.

    Args:
        user: User object
        thread_id: Optional thread ID from request
        creation_mode: Workflow creation mode

    Returns:
        Tuple of (session, thread_id, session_id)
    """
    if thread_id:
        session = await get_discovery_chat_session_by_thread_id(thread_id, user)
        if not session:
            session = await create_discovery_chat_session(
                user=user,
                thread_id=thread_id,
                workflow_creation_mode=creation_mode,
            )
        session_id = session.id
    else:
        thread_id = f"discovery-{uuid.uuid4().hex}"
        session = await create_discovery_chat_session(
            user=user,
            thread_id=thread_id,
            workflow_creation_mode=creation_mode,
        )
        session_id = session.id

    return session, thread_id, session_id


async def _setup_cost_tracking_context(user: User, thread_id: str) -> WorkflowRuntimeContext:
    """
    Setup cost tracking context for chat.

    Args:
        user: User object
        thread_id: Thread ID for the chat

    Returns:
        WorkflowRuntimeContext configured with user settings
    """
    # Get user settings for cost cap
    try:
        user_settings = await UserSettings.get(user=user)
        per_run_cost_cap_usd = user_settings.preferences.get("per_run_cost_cap_usd", 5.0)
    except DoesNotExist:
        per_run_cost_cap_usd = 5.0

    # Create runtime context for cost tracking
    runtime_context = WorkflowRuntimeContext(
        user=user,
        workflow_run_id=None,
        thread_id=thread_id,
        per_run_cost_cap_usd=per_run_cost_cap_usd,
        accumulated_cost_usd=0.0,
    )

    # Set context for callback access
    set_chat_runtime_context(runtime_context)

    return runtime_context


@router.post("/chat", response_model=ChatResponse)
async def discovery_chat_endpoint(  # pylint: disable=too-many-locals  # Reason: orchestration endpoint requires multiple local vars for clarity
    request: Request,
    chat_request: DiscoveryChatRequest,
) -> ChatResponse:
    """
    Discovery chat without a workflow (for workflow creation).

    The assistant can ask clarifying questions and create workflows based on user input.
    """
    logger.info("Discovery chat request: message_length=%d", len(chat_request.message))
    user = _require_user(request)

    # Get workflow creation mode
    creation_mode = (
        chat_request.workflow_creation_mode
        or await get_user_workflow_creation_mode(user)
    )

    model = chat_request.model or config.default_llm_model
    checkpointer = await get_checkpointer()

    # Get or create discovery session
    _session, thread_id, session_id = await _get_or_create_discovery_session(
        user=user,
        thread_id=chat_request.thread_id,
        creation_mode=creation_mode,
    )

    # Setup cost tracking context (sets global context as side effect)
    _runtime_context = await _setup_cost_tracking_context(user=user, thread_id=thread_id)

    # Set discovery mode context (no workflow state)
    set_user_for_thread(thread_id, user)

    # Create agent in discovery mode
    agent = create_nexus_chat_agent(
        model=model,
        checkpointer=checkpointer,
        workflow_state=None,  # No workflow in discovery mode
    )

    user_msg = HumanMessage(content=chat_request.message)

    # Track message
    await increment_chat_message_count(user)

    try:
        # Create cost tracking callback
        cost_callback = CostCapCallbackHandler()

        # Configure with callbacks
        config_dict = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [cost_callback],
        }

        # Initialize orchestrator
        orchestrator = ChatOrchestrator(
            agent=agent,
            checkpointer=checkpointer,
            health_service=CheckpointerHealthService(),
            detector=IncompleteToolCallDetector(),
            recovery_service=IncompleteToolCallRecoveryService(),
            reconnect_func=_recreate_checkpointer,
        )

        # Invoke agent
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

        # Extract response
        agent_messages = result.get("messages", []) if isinstance(result, dict) else []
        response_text = _extract_response_text(result)

        # Verify checkpoint
        if checkpointer and thread_id:
            await _verify_checkpoint_saved(checkpointer, thread_id)

        # Extract thinking
        thinking_steps = extract_thinking_from_messages(agent_messages)

        # Track assistant message
        await increment_chat_message_count(user)

        # Check if workflow was created (placeholder - will be implemented with agent specialist)
        workflow_created_id = None

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            thread_id=thread_id,
            thinking=thinking_steps if thinking_steps else None,
            interrupt_required=interrupt_required,
            interrupt_data=interrupt_data,
            workflow_created_id=workflow_created_id,
        )
    except RunCostCapExceeded as e:
        logger.warning(
            "Discovery chat cost cap exceeded",
            extra={
                "thread_id": thread_id,
                "accumulated_cost": e.accumulated_cost,
                "cost_cap": e.cost_cap,
            },
        )
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Cost cap exceeded",
            detail=e.to_dict(),
            status=402,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: API boundary, converting all exceptions to HTTP problem responses
        logger.error("Error in discovery chat: %s", e, exc_info=True)
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Discovery chat failed",
            detail=f"Failed to process discovery chat: {str(e)}",
            status=500
        )
    finally:
        clear_chat_runtime_context()
        clear_proposed_spec_for_thread(thread_id)
        clear_user_for_thread(thread_id)


@router.post("/{workflow_id}/chat", response_model=ChatResponse)
async def chat_with_workflow_endpoint(  # pylint: disable=too-many-locals # Reason: Complex endpoint orchestrating multiple services, requires refactoring to service layer
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
    workflow_state = await _prepare_workflow_state(workflow, chat_request.workflow_state)
    set_workflow_state_for_thread(thread_id, workflow_state)
    set_user_for_thread(thread_id, user)

    # Create agent
    agent = create_nexus_chat_agent(
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

    # Get user settings for max steps and cost cap
    try:
        user_settings = await UserSettings.get(user=user)
        max_agent_steps = user_settings.max_agent_steps or config.nexus_max_agent_steps
        per_run_cost_cap_usd = user_settings.preferences.get("per_run_cost_cap_usd", 5.0)
    except DoesNotExist:
        max_agent_steps = config.nexus_max_agent_steps
        per_run_cost_cap_usd = 5.0

    # Create runtime context for cost tracking
    runtime_context = WorkflowRuntimeContext(
        user=user,
        workflow_run_id=None,
        thread_id=thread_id,
        per_run_cost_cap_usd=per_run_cost_cap_usd,
        accumulated_cost_usd=0.0,
    )

    # Set context for callback access
    set_chat_runtime_context(runtime_context)

    try:
        # Create cost tracking callback
        cost_callback = CostCapCallbackHandler()

        # Configure with callbacks
        config_dict = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max_agent_steps,
            "callbacks": [cost_callback],
        }

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
    except RunCostCapExceeded as e:
        logger.warning(
            "Chat cost cap exceeded for thread '%s'",
            thread_id,
            extra={
                "thread_id": thread_id,
                "accumulated_cost": e.accumulated_cost,
                "cost_cap": e.cost_cap,
            },
        )
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Cost cap exceeded",
            detail=e.to_dict(),
            status=402,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: API boundary, converting all exceptions to HTTP problem responses
        # Handle other exceptions
        logger.error("Error in workflow chat: %s", e, exc_info=True)
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Chat processing failed",
            detail=f"Failed to process chat request: {str(e)}",
            status=500
        )
    finally:
        clear_chat_runtime_context()
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
async def resume_chat_endpoint(  # pylint: disable=too-many-locals,too-complex # Reason: Complex endpoint with resume logic, requires refactoring to service layer
    request: Request,
    workflow_id: str,
    resume_data: Dict[str, Any],
) -> ChatResponse:
    """
    Resume a chat session after an interrupt (e.g., clarification question).

    This endpoint handles resuming agent execution after a LangGraph interrupt.
    The resume_data should contain a Command object with resume information.
    """
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
    workflow_state = deepcopy(await workflow_state_snapshot(workflow))

    # Create agent
    agent = create_nexus_chat_agent(
        model=config.default_llm_model,
        checkpointer=checkpointer,
        workflow_state=workflow_state,
    )

    # Create Command object for resuming
    resume_command = Command(**command_data)

    # Get user settings for max steps
    try:
        user_settings = await UserSettings.get(user=user)
        max_agent_steps = user_settings.max_agent_steps or config.nexus_max_agent_steps
    except DoesNotExist:
        max_agent_steps = config.nexus_max_agent_steps

    # Resume agent execution
    config_dict = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": max_agent_steps,
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
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: API boundary, converting all exceptions to HTTP problem responses
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

    return WorkflowProposalActionResponse(
        proposal=WorkflowProposalPublic.model_validate(proposal, from_attributes=True),
        workflow_graph=await workflow_state_snapshot(workflow),
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

    return WorkflowProposalActionResponse(
        proposal=WorkflowProposalPublic.model_validate(proposal, from_attributes=True),
        workflow_graph=None,
    )


@router.get("/user/workflow-creation-mode", response_model=Dict[str, str])
async def get_user_creation_mode_endpoint(request: Request) -> Dict[str, str]:
    """Get user's default workflow creation mode."""
    user = _require_user(request)
    mode = await get_user_workflow_creation_mode(user)
    return {"mode": mode.value}


@router.post("/user/workflow-creation-mode")
async def update_user_creation_mode_endpoint(
    request: Request,
    body: Dict[str, str],
) -> Dict[str, str]:
    """Update user's default workflow creation mode."""
    user = _require_user(request)
    mode_str = body.get("mode")
    if not mode_str:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing mode",
            detail="mode is required",
            status=400
        )
    try:
        mode = WorkflowCreationMode(mode_str)
    except ValueError:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid mode",
            detail=f"Invalid mode: {mode_str}. Must be one of: AUTO_CREATE, ASK_FIRST, ON_ACCEPTANCE",
            status=400
        )
    await update_user_workflow_creation_mode(user, mode)
    return {"mode": mode.value}


__all__ = ["router"]
