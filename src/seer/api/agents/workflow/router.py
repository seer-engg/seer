# pylint: disable=too-many-lines
# Reason: Complex API router with multiple endpoints, requires architectural refactoring to split
"""
Workflow API router for CRUD and execution endpoints.
"""

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from seer.api.agents.checkpointer import get_checkpointer
from seer.api.core.errors import AUTH_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.api.core.middleware.organization import get_membership, get_organization
from seer.config import config
from seer.database import Organization, OrganizationMembership, User, UserPublic
from seer.logger import get_logger

from .chat_schema import (
    ChatMessage,
    ChatRequest,
    ChatResumeRequest,
    ChatSession,
    ChatSessionCreate,
    ChatSessionWithMessages,
    ClarificationAnswer,
    ClarificationAnswers,
    ClarificationQuestion,
    ClarificationQuestionOption,
    ClarificationQuestions,
    QuestionType,
    StreamEventType,
    WorkflowProposalActionResponse,
)
from .sse import get_stream_watermark, stream_events_sse

# Re-export for type hints
__all__ = ["router"]
from .chat_services import (
    SessionService,
)
from .models import WorkflowProposalPublic
from .services import (
    accept_workflow_proposal,
    create_chat_session,
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

router = APIRouter(prefix="/nexus", tags=["nexus"])


def _session_has_pending_interrupt(session: Any) -> bool:
    """Return True when the session is waiting for resume input."""
    return bool(
        session.current_execution_status == "interrupted"
        or session.pending_interrupt_data
        or session.pending_interrupt_type
    )


def _ensure_session_can_start_fresh_chat(session: Any) -> None:
    """Reject invalid fresh-chat transitions for interrupted or active sessions."""
    from seer.database.workflow_models import ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: avoid circular import at module load

    if session.current_execution_status in [
        ChatExecutionStatus.QUEUED,
        ChatExecutionStatus.RUNNING,
    ]:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Execution already in progress",
            detail="Cannot start new execution while another is in progress",
            status=409,
        )

    if _session_has_pending_interrupt(session):
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Session awaiting clarification",
            detail="This session is waiting for clarification. Use /chat/resume instead of starting a new /chat request.",
            status=409,
        )


def _ensure_session_can_resume(session: Any) -> None:
    """Reject invalid resume transitions when no interrupt is pending."""
    from seer.database.workflow_models import ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: avoid circular import at module load

    if session.current_execution_status in [
        ChatExecutionStatus.QUEUED,
        ChatExecutionStatus.RUNNING,
    ]:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Execution already in progress",
            detail="Cannot resume while another execution is still running",
            status=409,
        )

    if not _session_has_pending_interrupt(session):
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="No pending interrupt",
            detail="This session is not waiting for clarification. Start a new /chat request instead of /chat/resume.",
            status=409,
        )


def _require_user(request: Request) -> User:
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise_problem(
            type_uri=AUTH_PROBLEM,
            title="Unauthorized",
            detail="Unauthorized",
            status=401,
        )
    # Type guard: raise_problem raises an exception, so this will never execute if user is None
    assert user is not None
    return user


def _get_org_context(
    request: Request,
) -> tuple[Optional[Organization], Optional[OrganizationMembership]]:
    """Get optional organization context from request state."""
    try:
        org = get_organization(request)
        membership = get_membership(request)
        return org, membership
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: fallback to None if org context not available
        return None, None


def _transform_clarification_interrupt(interrupt_data: Dict[str, Any]) -> None:  # pylint: disable=unused-private-member # Reason: Reserved for future use in stream interrupt events
    """Transform clarification questions interrupt data for frontend."""
    if interrupt_data.get("type") != "clarification_questions":
        return

    questions_list = []
    for q in interrupt_data.get("questions", []):
        question_type = QuestionType(q["question_type"])

        # Build base question object
        question_kwargs = {
            "question_id": q["question_id"],
            "question": q["question"],
            "question_type": question_type,
            "reasoning": q.get("reasoning"),
        }

        if question_type == QuestionType.RESOURCE_PICKER:
            # Resource picker specific fields
            question_kwargs["provider"] = q.get("provider")
            question_kwargs["resource_type"] = q.get("resource_type")
            question_kwargs["display_field"] = q.get("display_field", "name")
            question_kwargs["value_field"] = q.get("value_field", "id")
            question_kwargs["search_enabled"] = q.get("search_enabled", True)
            question_kwargs["hierarchy"] = q.get("hierarchy", False)
            question_kwargs["depends_on"] = q.get("depends_on")
            question_kwargs["depends_on_field"] = q.get("depends_on_field")
            # Resource pickers don't have traditional options
            question_kwargs["options"] = []
        elif question_type == QuestionType.ACCOUNT_PICKER:
            # Account picker specific fields
            tool_name = q.get("tool_name")
            question_kwargs["tool_name"] = tool_name
            # Note: accounts list is populated by frontend via API call to /tools/{tool_name}/accounts
            # We pass tool_name so frontend knows which tool's accounts to fetch
            question_kwargs["accounts"] = None  # Frontend fetches dynamically
            question_kwargs[
                "options"
            ] = []  # Account pickers don't use traditional options
        else:
            # Choice type specific fields
            question_kwargs["options"] = [
                ClarificationQuestionOption(**opt) for opt in q.get("options", [])
            ]
            question_kwargs["min_selections"] = q.get("min_selections", 1)
            question_kwargs["max_selections"] = q.get("max_selections")

        question_obj = ClarificationQuestion(**question_kwargs)
        questions_list.append(question_obj.model_dump())

    questions_obj = ClarificationQuestions(
        questions=[ClarificationQuestion(**q) for q in questions_list]
    )
    interrupt_data["clarification_questions"] = questions_obj.model_dump()


def _validate_single_answer(
    answer: ClarificationAnswer, question_data: Dict[str, Any]
) -> None:
    """Validate a single answer against its question options."""
    question_type = question_data.get("question_type", "single_choice")

    # Resource picker answers don't have predefined options to validate against
    if question_type == QuestionType.RESOURCE_PICKER.value:
        # Just ensure at least one value is selected
        if not answer.selected_values:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Selection required",
                detail=f"At least one resource must be selected for question {answer.question_id}",
                status=400,
            )
        return

    # Account picker answers: selected_values contains connection_id
    if question_type == QuestionType.ACCOUNT_PICKER.value:
        if not answer.selected_values:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Account selection required",
                detail=f"An account must be selected for question {answer.question_id}",
                status=400,
            )
        # Validate that selected_values[0] is a valid integer connection_id
        try:
            int(answer.selected_values[0])
        except (ValueError, IndexError):
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Invalid account selection",
                detail=f"Invalid connection_id format for question {answer.question_id}",
                status=400,
            )
        return

    # Validate selected values for choice questions
    valid_values = {opt["value"] for opt in question_data.get("options", [])}
    invalid_selections = [v for v in answer.selected_values if v not in valid_values]

    if invalid_selections:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid selections",
            detail=f"Selected values not in available options: {invalid_selections}",
            status=400,
        )

    # Validate wildcard custom input
    wildcard_options = [
        opt for opt in question_data.get("options", []) if opt.get("is_wildcard")
    ]
    wildcard_values = {opt["value"] for opt in wildcard_options}
    has_wildcard_selection = any(v in wildcard_values for v in answer.selected_values)

    if has_wildcard_selection and not answer.custom_input:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Custom input required",
            detail=f"Custom input is required when selecting wildcard option for question {answer.question_id}",
            status=400,
        )


async def _validate_clarification_answers(
    checkpointer,
    thread_id: str,
    answers: ClarificationAnswers,
) -> None:
    """Validate clarification answers against original questions."""
    config_dict = {"configurable": {"thread_id": thread_id}}
    state_tuple = await checkpointer.aget_tuple(config_dict)

    if not state_tuple:
        return

    interrupt_payload = state_tuple.checkpoint.get("channel_values", {}).get(
        "__interrupt__"
    )
    if not interrupt_payload or not interrupt_payload[0]:
        return

    interrupt_data = interrupt_payload[0]
    if interrupt_data.get("type") != "clarification_questions":
        return

    questions_by_id = {q["question_id"]: q for q in interrupt_data.get("questions", [])}

    # Check all questions are answered
    answered_ids = {a.question_id for a in answers.answers}
    expected_ids = set(questions_by_id.keys())

    missing_answers = expected_ids - answered_ids
    if missing_answers:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Missing answers",
            detail=f"Missing answers for questions: {list(missing_answers)}",
            status=400,
        )

    extra_answers = answered_ids - expected_ids
    if extra_answers:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid question IDs",
            detail=f"Answers provided for unknown questions: {list(extra_answers)}",
            status=400,
        )

    # Validate each answer
    for ans in answers.answers:
        question_data = questions_by_id[ans.question_id]
        _validate_single_answer(ans, question_data)


async def _build_resume_command(
    resume_data: ChatResumeRequest,
    checkpointer,
    session_id: int,
) -> Command:
    """Build resume command from answers or raw command data."""
    resume_value: Any = None

    if resume_data.answers:
        # Clarification answers
        await _validate_clarification_answers(
            checkpointer, resume_data.thread_id, resume_data.answers
        )

        # Build list of answers for the agent
        resume_value = [
            {
                "question_id": ans.question_id,
                "selected_values": ans.selected_values,
                "custom_input": ans.custom_input,
            }
            for ans in resume_data.answers.answers
        ]

        # Build summary for chat message
        answer_summaries = []
        for ans in resume_data.answers.answers:
            summary = f"Q({ans.question_id}): {', '.join(ans.selected_values)}"
            if ans.custom_input:
                summary += f" (Custom: {ans.custom_input})"
            answer_summaries.append(summary)

        # Save user's answers to database
        await save_chat_message(
            session_id=session_id,
            role="user",
            content=f"Answered {len(resume_data.answers.answers)} questions:\n"
            + "\n".join(answer_summaries),
            metadata={"clarification_answers": resume_data.answers.model_dump()},
        )

    elif resume_data.message:
        resume_value = {
            "type": "free_text_response",
            "message": resume_data.message,
            "source": "chat_resume_message",
        }

        await save_chat_message(
            session_id=session_id,
            role="user",
            content=resume_data.message,
            metadata={"clarification_free_text": {"message": resume_data.message}},
        )

    elif resume_data.command:
        # Other interrupt types
        resume_value = resume_data.command.get("resume")

    else:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid resume data",
            detail="Exactly one of answers, message, or command must be provided",
            status=400,
        )

    return Command(resume=resume_value)


@router.post("/{workflow_id}/chat")
async def chat_with_workflow_endpoint(
    request: Request,
    workflow_id: str,
    chat_request: ChatRequest,
) -> StreamingResponse:
    """
    Chat with AI assistant about workflow.

    Returns a Server-Sent Events (SSE) stream of agent execution events.
    Events: session_info → agent_start → tool_start/end → agent_end → done

    To reconnect mid-stream, use GET /{workflow_id}/chat/stream/{session_id}
    with the Last-Event-ID header set to the last received Redis message ID.
    """
    from seer.agents.nexus.stream_publisher import StreamPublisher  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.database.workflow_models import ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: Only needed for status handling
    from seer.worker.tasks.chat import chat_execution_task  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    logger.info(
        "Chat request received: workflow_id=%s, message_length=%d, request_id=%s",
        workflow_id,
        len(chat_request.message),
        chat_request.request_id,
    )
    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(
        user, workflow_id, organization=org, membership=membership
    )

    # LLM credit limit check enforced by UsageLimitMiddleware
    model = chat_request.model or config.default_llm_model

    # Get or create session
    session, thread_id, session_id = await SessionService.get_or_create_session(
        workflow=workflow,
        user=user,
        thread_id=chat_request.thread_id,
        session_id=chat_request.session_id,
    )

    _ensure_session_can_start_fresh_chat(session)

    # Save current workflow state to session for UI/persistence
    session.current_workflow_state = await workflow_state_snapshot(workflow)
    await session.save(update_fields=["current_workflow_state"])

    # Save user message first (before async execution)
    await save_chat_message(
        session_id=session_id,
        role="user",
        content=chat_request.message,
    )

    execution_owner_id = uuid.uuid4().hex

    # Update session status to QUEUED
    session.current_execution_status = ChatExecutionStatus.QUEUED
    session.current_execution_task_id = execution_owner_id
    session.current_execution_error = None
    session.pending_interrupt_type = None
    session.pending_interrupt_data = None
    await session.save(
        update_fields=[
            "current_execution_status",
            "current_execution_task_id",
            "current_execution_error",
            "pending_interrupt_type",
            "pending_interrupt_data",
        ]
    )

    # Enqueue task
    task = await chat_execution_task.kiq(
        session_id=session_id,
        user_id=user.id,
        message=chat_request.message,
        workflow_id=workflow.id,
        model=model,
        execution_task_id=execution_owner_id,
        user_timezone=chat_request.timezone,
    )

    logger.info(
        "Chat execution enqueued",
        extra={
            "session_id": session_id,
            "task_id": task.task_id,
            "execution_owner_id": execution_owner_id,
            "request_id": chat_request.request_id,
        },
    )

    # Snapshot stream position *before* publishing any new events so the
    # client only receives new events, not a replay of the entire session
    # history (which would re-trigger the clarification questions UI).
    watermark_id = await get_stream_watermark(session_id)

    # Pre-publish SESSION_INFO so client immediately gets session_id + thread_id
    # before the worker picks up the task
    publisher = StreamPublisher(session_id)
    await publisher.publish(
        StreamEventType.SESSION_INFO,
        {
            "session_id": session_id,
            "thread_id": thread_id,
            "execution_task_id": execution_owner_id,
        },
    )

    return StreamingResponse(
        stream_events_sse(session_id, last_event_id=watermark_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/{workflow_id}/chat/stream/{session_id}")
async def stream_chat_events_endpoint(
    request: Request,
    workflow_id: str,
    session_id: int,
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """
    Reconnect to an existing chat SSE stream.

    Used by clients that disconnected mid-run. Pass the Last-Event-ID header
    with the last Redis Stream message ID received — the stream will resume
    from that point without replaying earlier events.

    If the stream TTL has expired and the session is COMPLETED, falls back to
    DB message history.
    """
    user = _require_user(request)
    workflow = await get_workflow(user, workflow_id)
    await get_chat_session(session_id, workflow)  # Auth check

    return StreamingResponse(
        stream_events_sse(session_id, last_event_id=last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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
    org, membership = _get_org_context(request)
    workflow = await get_workflow(
        user, workflow_id, organization=org, membership=membership
    )

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
        current_execution_status=session.current_execution_status.value
        if session.current_execution_status
        else None,
        current_execution_task_id=session.current_execution_task_id,
        pending_interrupt_type=session.pending_interrupt_type,
        pending_interrupt_data=session.pending_interrupt_data,
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
    org, membership = _get_org_context(request)
    workflow = await get_workflow(
        user, workflow_id, organization=org, membership=membership
    )
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
            current_execution_status=session.current_execution_status.value
            if session.current_execution_status
            else None,
            current_execution_task_id=session.current_execution_task_id,
            pending_interrupt_type=session.pending_interrupt_type,
            pending_interrupt_data=session.pending_interrupt_data,
        )
        for session in sessions
    ]


@router.get(
    "/{workflow_id}/chat/sessions/{session_id}", response_model=ChatSessionWithMessages
)
async def get_chat_session_endpoint(
    request: Request,
    workflow_id: str,
    session_id: int,
) -> ChatSessionWithMessages:
    """Get a chat session with its messages."""
    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(
        user, workflow_id, organization=org, membership=membership
    )
    session = await get_chat_session(session_id, workflow)

    messages = await load_chat_history(session_id)

    messages_out = [
        ChatMessage(
            id=msg.id,
            session_id=session_id,  # Use the session_id parameter directly
            role=msg.role,
            content=msg.content,
            thinking=msg.thinking,
            suggested_edits=msg.suggested_edits,
            proposal=WorkflowProposalPublic.model_validate(
                msg.proposal, from_attributes=True
            )
            if msg.proposal
            else None,
            metadata=msg.metadata,
            created_at=msg.created_at,
        )
        for msg in messages
    ]

    # If the session is waiting for clarification answers, annotate the last assistant message
    # so the frontend can restore the clarification card from message history (reconnect or history navigation).
    # pending_interrupt_data is cleared by the worker when the user answers, so this is safe.
    if session.pending_interrupt_data:
        for msg_out in reversed(messages_out):
            if msg_out.role == "assistant":
                msg_out.interrupt_required = True
                msg_out.interrupt_data = session.pending_interrupt_data
                break

    return ChatSessionWithMessages(
        id=session.id,
        workflow_id=workflow.workflow_id,
        user=UserPublic.model_validate(session.user, from_attributes=True),
        thread_id=session.thread_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        current_execution_status=session.current_execution_status.value
        if session.current_execution_status
        else None,
        current_execution_task_id=session.current_execution_task_id,
        pending_interrupt_type=session.pending_interrupt_type,
        pending_interrupt_data=session.pending_interrupt_data,
        messages=messages_out,
    )


@router.post("/{workflow_id}/chat/resume")
async def resume_chat_endpoint(
    request: Request,
    workflow_id: str,
    resume_data: ChatResumeRequest,
) -> StreamingResponse:
    """
    Resume chat after interrupt (clarification question or other interrupt type).

    For clarification questions, provide 'answers' with structured selections or
    'message' with a free-text reply.
    For other interrupts, provide 'command' with raw Command data.

    Returns an SSE stream — same format as POST /{workflow_id}/chat.
    """
    from seer.agents.nexus.stream_publisher import StreamPublisher  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.worker.tasks.chat import chat_resume_task  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.database.workflow_models import ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: Only needed for status handling

    logger.info(
        "Resume request received: workflow_id=%s, thread_id=%s, request_id=%s",
        workflow_id,
        resume_data.thread_id,
        resume_data.request_id,
    )
    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(
        user, workflow_id, organization=org, membership=membership
    )

    # Get checkpointer and session
    checkpointer = await get_checkpointer()
    session = await get_chat_session_by_thread_id(resume_data.thread_id, workflow)
    if not session:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Session not found",
            detail=f"Chat session not found for thread_id: {resume_data.thread_id}",
            status=404,
        )

    # Type guard: raise_problem raises an exception, so session is guaranteed to be not None here
    assert session is not None
    session_id = session.id

    _ensure_session_can_resume(session)

    # Build resume command (validates answer)
    resume_command = await _build_resume_command(resume_data, checkpointer, session_id)

    execution_owner_id = uuid.uuid4().hex

    # Update session status to QUEUED
    session.current_execution_status = ChatExecutionStatus.QUEUED
    session.current_execution_task_id = execution_owner_id
    session.current_execution_error = None
    session.pending_interrupt_type = None
    session.pending_interrupt_data = None
    await session.save(
        update_fields=[
            "current_execution_status",
            "current_execution_task_id",
            "current_execution_error",
            "pending_interrupt_type",
            "pending_interrupt_data",
        ]
    )

    # Enqueue task
    task = await chat_resume_task.kiq(
        session_id=session_id,
        user_id=user.id,
        thread_id=resume_data.thread_id,
        resume_command_data=resume_command.resume,
        workflow_id=workflow.id,
        execution_task_id=execution_owner_id,
    )

    logger.info(
        "Chat resume enqueued",
        extra={
            "session_id": session_id,
            "task_id": task.task_id,
            "execution_owner_id": execution_owner_id,
            "request_id": resume_data.request_id,
        },
    )

    # Snapshot the stream position *before* publishing any new events.
    # Passing this as last_event_id tells stream_events_sse to start from
    # after the existing history, so the client only receives new events
    # (SESSION_INFO + agent output) rather than a replay of the entire
    # prior run (which would re-trigger the clarification questions UI).
    watermark_id = await get_stream_watermark(session_id)

    # Pre-publish SESSION_INFO so client immediately gets session_id + thread_id
    publisher = StreamPublisher(session_id)
    await publisher.publish(
        StreamEventType.SESSION_INFO,
        {
            "session_id": session_id,
            "thread_id": resume_data.thread_id,
            "execution_task_id": execution_owner_id,
        },
    )

    return StreamingResponse(
        stream_events_sse(session_id, last_event_id=watermark_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/{workflow_id}/proposals/{proposal_id}", response_model=WorkflowProposalPublic
)
async def get_proposal_endpoint(
    request: Request,
    workflow_id: str,
    proposal_id: int,
) -> WorkflowProposalPublic:
    """Fetch a single workflow proposal."""
    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(
        user, workflow_id, organization=org, membership=membership
    )
    proposal = await get_workflow_proposal(workflow, proposal_id)
    await proposal.fetch_related("created_by", "workflow", "session")
    return WorkflowProposalPublic.model_validate(proposal, from_attributes=True)


@router.post(
    "/{workflow_id}/proposals/{proposal_id}/accept",
    response_model=WorkflowProposalActionResponse,
)
async def accept_proposal_endpoint(
    request: Request,
    workflow_id: str,
    proposal_id: int,
) -> WorkflowProposalActionResponse:
    """Accept a workflow proposal and apply its changes."""
    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(
        user, workflow_id, organization=org, membership=membership
    )
    proposal, workflow = await accept_workflow_proposal(
        workflow,
        proposal_id,
        actor=user,
    )
    await proposal.fetch_related("created_by", "workflow", "session")

    return WorkflowProposalActionResponse(
        proposal=WorkflowProposalPublic.model_validate(proposal, from_attributes=True),
        workflow_graph=await workflow_state_snapshot(workflow),
    )


@router.post(
    "/{workflow_id}/proposals/{proposal_id}/reject",
    response_model=WorkflowProposalActionResponse,
)
async def reject_proposal_endpoint(
    request: Request,
    workflow_id: str,
    proposal_id: int,
) -> WorkflowProposalActionResponse:
    """Reject a workflow proposal without applying changes."""
    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(
        user, workflow_id, organization=org, membership=membership
    )
    proposal = await reject_workflow_proposal(workflow, proposal_id)
    await proposal.fetch_related("created_by", "workflow", "session")

    return WorkflowProposalActionResponse(
        proposal=WorkflowProposalPublic.model_validate(proposal, from_attributes=True),
        workflow_graph=None,
    )


__all__ = ["router"]
