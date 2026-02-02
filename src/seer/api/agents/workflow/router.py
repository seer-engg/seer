# pylint: disable=too-many-lines,duplicate-code
# Reason: Complex API router with multiple endpoints, requires architectural refactoring to split;
# shared helper functions duplicated in worker tasks for background execution
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
    create_nexus_chat_agent,
    extract_thinking_from_messages,
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
from seer.database.workflow_models import WorkflowCreationMode
from seer.logger import get_logger
from seer.observability import (
    increment_chat_message_count,
)
from seer.observability.exceptions import RunCostCapExceeded

from .chat_schema import (
    ChatMessage,
    ChatRequest,
    ChatResumeRequest,
    ChatResponse,
    ChatSession,
    ChatSessionCreate,
    ChatSessionWithMessages,
    ChatStatusResponse,
    ClarificationQuestion,
    ClarificationQuestionOption,
    QuestionType,
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
    # Type guard: raise_problem raises an exception, so this will never execute if user is None
    assert user is not None
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


async def _get_user_settings_and_context(
    user: User,
    thread_id: str,
) -> Tuple[int, WorkflowRuntimeContext]:
    """Get user settings and create runtime context for cost tracking."""
    try:
        user_settings = await UserSettings.get(user=user)
        max_agent_steps = user_settings.max_agent_steps or config.nexus_max_agent_steps
        per_run_cost_cap_usd = user_settings.preferences.get("per_run_cost_cap_usd", 5.0)
    except DoesNotExist:
        max_agent_steps = config.nexus_max_agent_steps
        per_run_cost_cap_usd = 5.0

    runtime_context = WorkflowRuntimeContext(
        user=user,
        workflow_run_id=None,
        thread_id=thread_id,
        per_run_cost_cap_usd=per_run_cost_cap_usd,
        accumulated_cost_usd=0.0,
    )

    return max_agent_steps, runtime_context


def _transform_clarification_interrupt(interrupt_data: Dict[str, Any]) -> None:
    """Transform clarification question interrupt data for frontend."""
    if interrupt_data.get("type") == "clarification_question":
        question_obj = ClarificationQuestion(
            question_id=interrupt_data["question_id"],
            question=interrupt_data["question"],
            question_type=QuestionType(interrupt_data["question_type"]),
            options=[
                ClarificationQuestionOption(**opt)
                for opt in interrupt_data["options"]
            ],
            min_selections=interrupt_data.get("min_selections", 1),
            max_selections=interrupt_data.get("max_selections"),
        )
        interrupt_data["clarification_question"] = question_obj.model_dump()


async def _save_response_and_get_proposal(
    session_id: int,
    thread_id: str,
    response_text: str,
    thinking_steps: list[str],
    user: User,
) -> Tuple[Optional[Any], Optional[WorkflowProposalPublic], Optional[str]]:
    """Save assistant message and retrieve any pending proposal."""
    from seer.database import WorkflowProposal  # pylint: disable=import-outside-toplevel # Reason: Only needed in this code path

    proposal = await WorkflowProposal.get_or_none(
        thread_id=thread_id,
        status=WorkflowProposal.STATUS_PENDING
    ).prefetch_related('created_by', 'workflow', 'session')

    proposal_public = None
    proposal_error = None
    if proposal:
        proposal_public = WorkflowProposalPublic.model_validate(proposal, from_attributes=True)

    await save_chat_message(
        session_id=session_id,
        role="assistant",
        content=response_text,
        thinking="\n".join(thinking_steps) if thinking_steps else None,
        suggested_edits=proposal.spec if proposal else None,
        proposal=proposal,
    )

    await increment_chat_message_count(user)

    return proposal, proposal_public, proposal_error


async def _invoke_agent_with_orchestrator(
    agent,
    checkpointer,
    user_msg: HumanMessage,
    thread_id: str,
    max_agent_steps: int,
) -> Dict[str, Any]:
    """Invoke agent using orchestrator with health checks."""
    cost_callback = CostCapCallbackHandler()

    config_dict = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_agent_steps,
        "callbacks": [cost_callback],
    }

    orchestrator = ChatOrchestrator(
        agent=agent,
        checkpointer=checkpointer,
        health_service=CheckpointerHealthService(),
        detector=IncompleteToolCallDetector(),
        recovery_service=IncompleteToolCallRecoveryService(),
        reconnect_func=_recreate_checkpointer,
    )

    return await orchestrator.invoke_with_health_checks(user_msg, config_dict)


def _detect_and_transform_interrupts(
    result: Dict[str, Any],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Detect interrupts from result and transform clarification questions."""
    interrupt_required, interrupt_data = InterruptHandler.extract_interrupt_from_result(result)

    if interrupt_required and interrupt_data:
        _transform_clarification_interrupt(interrupt_data)

    return interrupt_required, interrupt_data


async def _process_agent_result(
    result: Dict[str, Any],
    checkpointer,
    thread_id: str,
) -> Tuple[str, list[str]]:
    """Extract response text, thinking steps, and verify checkpoint."""
    agent_messages = result.get("messages", []) if isinstance(result, dict) else []
    response_text = _extract_response_text(result)
    thinking_steps = extract_thinking_from_messages(agent_messages)

    logger.info(
        "Agent completed for thread %s, response_length=%d",
        thread_id,
        len(response_text)
    )

    if checkpointer and thread_id:
        await _verify_checkpoint_saved(checkpointer, thread_id)

    return response_text, thinking_steps


async def _validate_clarification_answer(
    checkpointer,
    thread_id: str,
    answer,
) -> None:
    """Validate clarification answer against original question options."""
    config_dict = {"configurable": {"thread_id": thread_id}}
    state_tuple = await checkpointer.aget_tuple(config_dict)

    if not state_tuple:
        return

    interrupt_payload = state_tuple.checkpoint.get("channel_values", {}).get("__interrupt__")
    if not interrupt_payload or not interrupt_payload[0]:
        return

    interrupt_data = interrupt_payload[0]
    if interrupt_data.get("type") != "clarification_question":
        return

    # Validate selected values
    valid_values = {opt["value"] for opt in interrupt_data["options"]}
    invalid_selections = [v for v in answer.selected_values if v not in valid_values]

    if invalid_selections:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid selections",
            detail=f"Selected values not in available options: {invalid_selections}",
            status=400
        )

    # Validate wildcard custom input
    wildcard_options = [opt for opt in interrupt_data["options"] if opt.get("is_wildcard")]
    wildcard_values = {opt["value"] for opt in wildcard_options}
    has_wildcard_selection = any(v in wildcard_values for v in answer.selected_values)

    if has_wildcard_selection and not answer.custom_input:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Custom input required",
            detail="Custom input is required when selecting wildcard option",
            status=400
        )


async def _build_resume_command(
    resume_data: ChatResumeRequest,
    checkpointer,
    session_id: int,
) -> Command:
    """Build resume command from answer or raw command data."""
    resume_value: Any = None

    if resume_data.answer:
        # Validate and build clarification answer
        await _validate_clarification_answer(checkpointer, resume_data.thread_id, resume_data.answer)

        resume_value = {
            "selected_values": resume_data.answer.selected_values,
            "custom_input": resume_data.answer.custom_input,
        }

        # Save user's answer to database
        await save_chat_message(
            session_id=session_id,
            role="user",
            content=f"Selected: {', '.join(resume_data.answer.selected_values)}" +
                    (f" (Custom: {resume_data.answer.custom_input})" if resume_data.answer.custom_input else ""),
            metadata={"clarification_answer": resume_data.answer.model_dump()},
        )
    elif resume_data.command:
        # Other interrupt types
        resume_value = resume_data.command.get("resume")
    else:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid resume data",
            detail="Either answer or command must be provided",
            status=400
        )

    return Command(resume=resume_value)


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


@router.post("/{workflow_id}/chat", response_model=ChatResponse)
async def chat_with_workflow_endpoint(  # pylint: disable=too-many-locals # Reason: Complex endpoint orchestrating multiple services, requires refactoring to service layer
    request: Request,
    workflow_id: str,
    chat_request: ChatRequest,
    async_mode: bool = Query(
        default=True,
        description="Execute in background. Client must poll /chat/status/{session_id} for completion. Set to false for synchronous execution."
    ),
) -> ChatResponse:
    """
    Chat with AI assistant about workflow.

    The assistant can analyze the workflow and suggest edits.
    Supports session persistence and human-in-the-loop interrupts.

    When async_mode=True (default), execution runs in background and client must poll
    /chat/status/{session_id} for results. This makes the API resilient to client disconnections.
    """
    logger.info("Chat request received: workflow_id=%s, message_length=%d, async_mode=%s", workflow_id, len(chat_request.message), async_mode)
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

    # Check if session already has execution in progress
    from seer.database.workflow_models import ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: Only needed for async mode
    if session.current_execution_status in [ChatExecutionStatus.QUEUED, ChatExecutionStatus.RUNNING]:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Execution already in progress",
            detail="Cannot start new execution while another is in progress",
            status=409
        )

    # Prepare workflow state and save to session
    workflow_state = await _prepare_workflow_state(workflow, chat_request.workflow_state)
    session.current_workflow_state = workflow_state
    await session.save(update_fields=['current_workflow_state'])

    # Save user message first (before async execution)
    await save_chat_message(
        session_id=session_id,
        role="user",
        content=chat_request.message,
    )

    # Track user message (global count, not per-workflow)
    await increment_chat_message_count(user)

    # Handle async mode
    if async_mode:
        # Enqueue background task
        from seer.worker.tasks.chat import chat_execution_task  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

        # Update session status to QUEUED
        session.current_execution_status = ChatExecutionStatus.QUEUED
        session.current_execution_error = None
        session.pending_interrupt_type = None
        session.pending_interrupt_data = None
        await session.save(update_fields=[
            'current_execution_status',
            'current_execution_error',
            'pending_interrupt_type',
            'pending_interrupt_data',
        ])

        # Enqueue task
        task = await chat_execution_task.kiq(
            session_id=session_id,
            user_id=user.id,
            message=chat_request.message,
            workflow_id=workflow.id,
            workflow_state=workflow_state,
            model=model,
        )

        # Save task ID to session
        session.current_execution_task_id = task.task_id
        await session.save(update_fields=['current_execution_task_id'])

        logger.info(
            "Chat execution enqueued",
            extra={
                "session_id": session_id,
                "task_id": task.task_id,
            }
        )

        # Return minimal response for async mode
        return ChatResponse(
            response="",
            session_id=session_id,
            thread_id=thread_id,
            execution_status=ChatExecutionStatus.QUEUED.value,
            execution_task_id=task.task_id,
        )

    # Synchronous mode (original behavior)
    # Create agent
    agent = create_nexus_chat_agent(
        model=model,
        checkpointer=checkpointer,
        workflow_state=workflow_state,
    )

    user_msg = HumanMessage(content=chat_request.message)

    # Get user settings and create runtime context
    max_agent_steps, runtime_context = await _get_user_settings_and_context(user, thread_id)

    # Set context for callback access
    set_chat_runtime_context(runtime_context)

    try:
        # Invoke agent and get result
        result = await _invoke_agent_with_orchestrator(
            agent, checkpointer, user_msg, thread_id, max_agent_steps
        )

        # Detect and transform interrupts
        interrupt_required, interrupt_data = _detect_and_transform_interrupts(result)

        # Extract response and verify checkpoint
        response_text, thinking_steps = await _process_agent_result(
            result, checkpointer, thread_id
        )

        # Save response and get proposal
        _, proposal_public, proposal_error = await _save_response_and_get_proposal(
            session_id, thread_id, response_text, thinking_steps, user
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
            detail=str(e.to_dict()),
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


@router.get("/{workflow_id}/chat/status/{session_id}", response_model=ChatStatusResponse)
async def get_chat_status_endpoint(
    request: Request,
    workflow_id: str,
    session_id: int,
) -> ChatStatusResponse:
    """
    Poll for chat execution status.

    Returns execution status and results when completed.
    Includes interrupt data if agent requires clarification.
    """
    user = _require_user(request)
    workflow = await get_workflow(user, workflow_id)
    session = await get_chat_session(session_id, workflow)

    # Get latest assistant message for completed executions
    response_text = None
    thinking_steps = None
    proposal_public = None

    from seer.database.workflow_models import ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: Only needed for status checks
    from seer.database import WorkflowChatMessage  # pylint: disable=import-outside-toplevel # Reason: Only needed for status endpoint

    if session.current_execution_status == ChatExecutionStatus.COMPLETED:
        # Fetch latest assistant message
        latest_message = await WorkflowChatMessage.filter(
            session_id=session_id,
            role="assistant"
        ).order_by('-created_at').first().prefetch_related('proposal')

        if latest_message:
            response_text = latest_message.content
            thinking_steps = latest_message.thinking.split("\n") if latest_message.thinking else None

            # Get proposal if any
            if latest_message.proposal:
                await latest_message.proposal.fetch_related('created_by', 'workflow', 'session')
                proposal_public = WorkflowProposalPublic.model_validate(latest_message.proposal, from_attributes=True)

    # Transform interrupt data for frontend
    interrupt_data = None
    if session.pending_interrupt_data:
        interrupt_data = session.pending_interrupt_data
        _transform_clarification_interrupt(interrupt_data)

    return ChatStatusResponse(
        status=session.current_execution_status.value if session.current_execution_status else "unknown",
        session_id=session_id,
        thread_id=session.thread_id,
        response=response_text,
        thinking=thinking_steps,
        proposal=proposal_public,
        interrupt_required=session.current_execution_status == ChatExecutionStatus.INTERRUPTED,
        interrupt_data=interrupt_data,
        error=session.current_execution_error,
        started_at=session.current_execution_started_at,
        finished_at=session.current_execution_finished_at,
    )


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


@router.post("/{workflow_id}/chat/resume", response_model=ChatResponse)
async def resume_chat_endpoint(  # pylint: disable=too-many-locals # Reason: Complex endpoint with resume logic, requires refactoring to service layer
    request: Request,
    workflow_id: str,
    resume_data: ChatResumeRequest,
    async_mode: bool = Query(
        default=True,
        description="Execute in background. Client must poll /chat/status/{session_id} for completion. Set to false for synchronous execution."
    ),
) -> ChatResponse:
    """
    Resume chat after interrupt (clarification question or other interrupt type).

    For clarification questions, provide 'answer' with selected values.
    For other interrupts, provide 'command' with raw Command data.

    When async_mode=True (default), execution runs in background and client must poll
    /chat/status/{session_id} for results.
    """
    logger.info("Resume request received: workflow_id=%s, thread_id=%s, async_mode=%s", workflow_id, resume_data.thread_id, async_mode)
    user = _require_user(request)
    workflow = await get_workflow(user, workflow_id)

    # Get checkpointer and session
    checkpointer = await get_checkpointer()
    session = await get_chat_session_by_thread_id(resume_data.thread_id, workflow)
    if not session:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Session not found",
            detail=f"Chat session not found for thread_id: {resume_data.thread_id}",
            status=404
        )

    # Type guard: raise_problem raises an exception, so session is guaranteed to be not None here
    assert session is not None
    session_id = session.id

    # Get current workflow state
    workflow_state = deepcopy(await workflow_state_snapshot(workflow))

    # Build resume command (validates answer)
    resume_command = await _build_resume_command(resume_data, checkpointer, session_id)

    # Handle async mode
    if async_mode:
        # Enqueue background task
        from seer.worker.tasks.chat import chat_resume_task  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.database.workflow_models import ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: Only needed for async mode

        # Update session status to QUEUED
        session.current_execution_status = ChatExecutionStatus.QUEUED
        session.current_execution_error = None
        session.pending_interrupt_type = None
        session.pending_interrupt_data = None
        await session.save(update_fields=[
            'current_execution_status',
            'current_execution_error',
            'pending_interrupt_type',
            'pending_interrupt_data',
        ])

        # Enqueue task
        task = await chat_resume_task.kiq(
            session_id=session_id,
            user_id=user.id,
            thread_id=resume_data.thread_id,
            resume_command_data=resume_command.resume,
            workflow_id=workflow.id,
            workflow_state=workflow_state,
        )

        # Save task ID to session
        session.current_execution_task_id = task.task_id
        await session.save(update_fields=['current_execution_task_id'])

        logger.info(
            "Chat resume enqueued",
            extra={
                "session_id": session_id,
                "task_id": task.task_id,
            }
        )

        # Return minimal response for async mode
        return ChatResponse(
            response="",
            session_id=session_id,
            thread_id=resume_data.thread_id,
            execution_status=ChatExecutionStatus.QUEUED.value,
            execution_task_id=task.task_id,
        )

    # Synchronous mode (original behavior)
    agent = create_nexus_chat_agent(
        model=config.default_llm_model,
        checkpointer=checkpointer,
        workflow_state=workflow_state,
    )

    # Get user settings
    try:
        user_settings = await UserSettings.get(user=user)
        max_agent_steps = user_settings.max_agent_steps or config.nexus_max_agent_steps
    except DoesNotExist:
        max_agent_steps = config.nexus_max_agent_steps

    # Resume agent execution
    config_dict = {
        "configurable": {"thread_id": resume_data.thread_id},
        "recursion_limit": max_agent_steps,
    }

    # Set thread_id in context variable
    token = _current_thread_id.set(resume_data.thread_id)
    try:
        result = await agent.ainvoke(resume_command, config=config_dict)

        # Extract response and thinking
        agent_messages = result.get("messages", [])
        response_text = _extract_response_text(result) if agent_messages else "Continuing..."
        thinking_steps = extract_thinking_from_messages(agent_messages)

        # Detect and transform interrupts
        interrupt_required, interrupt_data = InterruptHandler.extract_interrupt_from_result(result)
        if interrupt_required and interrupt_data:
            _transform_clarification_interrupt(interrupt_data)

        # Get proposal and save response
        from seer.database import WorkflowProposal  # pylint: disable=import-outside-toplevel # Reason: Only needed in this code path
        proposal = await WorkflowProposal.get_or_none(
            thread_id=resume_data.thread_id,
            status=WorkflowProposal.STATUS_PENDING
        ).prefetch_related('created_by', 'workflow', 'session')

        proposal_public = None
        proposal_error = None
        if proposal:
            proposal_public = WorkflowProposalPublic.model_validate(proposal, from_attributes=True)

        # Save assistant message
        await save_chat_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            thinking="\n".join(thinking_steps) if thinking_steps else None,
            suggested_edits=proposal.spec if proposal else None,
            proposal=proposal,
        )

        return ChatResponse(
            response=response_text,
            proposal=proposal_public,
            proposal_error=proposal_error,
            session_id=session_id,
            thread_id=resume_data.thread_id,
            thinking=thinking_steps if thinking_steps else None,
            interrupt_required=interrupt_required,
            interrupt_data=interrupt_data,
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
        _current_thread_id.reset(token)


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
        # Use database enum for the service call
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
