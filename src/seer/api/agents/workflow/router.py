# pylint: disable=too-many-lines
# Reason: Complex API router with multiple endpoints, requires architectural refactoring to split
"""
Workflow API router for CRUD and execution endpoints.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from seer.api.agents.checkpointer import get_checkpointer
from seer.api.core.errors import AUTH_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.api.core.middleware.organization import get_membership, get_organization
from seer.config import config
from seer.database import Organization, OrganizationMembership, User, UserPublic
from seer.database.workflow_models import WorkflowCreationMode
from seer.logger import get_logger

from .chat_schema import (
    ChatMessage,
    ChatRequest,
    ChatResumeRequest,
    ChatResponse,
    ChatSession,
    ChatSessionCreate,
    ChatSessionWithMessages,
    ChatStatusResponse,
    ClarificationAnswer,
    ClarificationAnswers,
    ClarificationQuestion,
    ClarificationQuestionOption,
    ClarificationQuestions,
    QuestionType,
    WorkflowProposalActionResponse,
)

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


def _get_org_context(request: Request) -> tuple[Optional[Organization], Optional[OrganizationMembership]]:
    """Get optional organization context from request state."""
    try:
        org = get_organization(request)
        membership = get_membership(request)
        return org, membership
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: fallback to None if org context not available
        return None, None


def _transform_clarification_interrupt(interrupt_data: Dict[str, Any]) -> None:
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
            question_kwargs["options"] = []  # Account pickers don't use traditional options
        else:
            # Choice type specific fields
            question_kwargs["options"] = [
                ClarificationQuestionOption(**opt)
                for opt in q.get("options", [])
            ]
            question_kwargs["min_selections"] = q.get("min_selections", 1)
            question_kwargs["max_selections"] = q.get("max_selections")

        question_obj = ClarificationQuestion(**question_kwargs)
        questions_list.append(question_obj.model_dump())

    questions_obj = ClarificationQuestions(
        questions=[ClarificationQuestion(**q) for q in questions_list]
    )
    interrupt_data["clarification_questions"] = questions_obj.model_dump()


def _validate_single_answer(answer: ClarificationAnswer, question_data: Dict[str, Any]) -> None:
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
                status=400
            )
        return

    # Account picker answers: selected_values contains connection_id
    if question_type == QuestionType.ACCOUNT_PICKER.value:
        if not answer.selected_values:
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Account selection required",
                detail=f"An account must be selected for question {answer.question_id}",
                status=400
            )
        # Validate that selected_values[0] is a valid integer connection_id
        try:
            int(answer.selected_values[0])
        except (ValueError, IndexError):
            raise_problem(
                type_uri=VALIDATION_PROBLEM,
                title="Invalid account selection",
                detail=f"Invalid connection_id format for question {answer.question_id}",
                status=400
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
            status=400
        )

    # Validate wildcard custom input
    wildcard_options = [opt for opt in question_data.get("options", []) if opt.get("is_wildcard")]
    wildcard_values = {opt["value"] for opt in wildcard_options}
    has_wildcard_selection = any(v in wildcard_values for v in answer.selected_values)

    if has_wildcard_selection and not answer.custom_input:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Custom input required",
            detail=f"Custom input is required when selecting wildcard option for question {answer.question_id}",
            status=400
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

    interrupt_payload = state_tuple.checkpoint.get("channel_values", {}).get("__interrupt__")
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
            status=400
        )

    extra_answers = answered_ids - expected_ids
    if extra_answers:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid question IDs",
            detail=f"Answers provided for unknown questions: {list(extra_answers)}",
            status=400
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
            content=f"Answered {len(resume_data.answers.answers)} questions:\n" + "\n".join(answer_summaries),
            metadata={"clarification_answers": resume_data.answers.model_dump()},
        )

    elif resume_data.command:
        # Other interrupt types
        resume_value = resume_data.command.get("resume")

    else:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid resume data",
            detail="Either answers or command must be provided",
            status=400
        )

    return Command(resume=resume_value)


async def _handle_recovery_stream(  # pylint: disable=too-many-positional-arguments # Reason: All params needed
    orchestrator, message: str, config_dict, thread_id: str, session, session_id: int,
):
    """Handle the incomplete-tool-call recovery path and yield SSE events."""
    from seer.agents.nexus import extract_thinking_from_messages  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.worker.tasks.chat import _handle_agent_result  # pylint: disable=import-outside-toplevel # Reason: Reuse shared logic
    from seer.api.agents.workflow.chat_services import InterruptHandler  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.api.agents.workflow.streaming import build_final_sse_event  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    user_msg = HumanMessage(content=message)
    result = await orchestrator._recover_from_incomplete_state(user_msg, config_dict, thread_id)  # pylint: disable=protected-access # Reason: Internal module helper
    response_text = result.get("messages", [])[-1].content if result.get("messages") else ""
    yield f"data: {json.dumps({'token': response_text})}\n\n"

    agent_messages = result.get("messages", [])
    thinking_steps = extract_thinking_from_messages(agent_messages)
    interrupt_required, interrupt_data = InterruptHandler.extract_interrupt_from_result(result)
    await _handle_agent_result(session, result, thread_id, session_id)

    proposal = None
    if not interrupt_required:
        from seer.database.workflow_models import WorkflowProposal  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        proposal = await WorkflowProposal.get_or_none(thread_id=thread_id, status=WorkflowProposal.STATUS_PENDING)
        if proposal:
            await proposal.fetch_related('created_by', 'workflow', 'session')

    yield build_final_sse_event(session_id, thread_id, thinking_steps, interrupt_required, interrupt_data, proposal)


@router.post("/{workflow_id}/chat/stream")
async def stream_chat_with_workflow_endpoint(  # pylint: disable=too-many-locals,too-many-statements,too-complex # Reason: SSE endpoint with nested generator
    request: Request,
    workflow_id: str,
    chat_request: ChatRequest,
):
    """Stream chat response via SSE. Runs agent inline with token-level streaming."""
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.database.workflow_models import ChatExecutionStatus
    from seer.agents.nexus import create_nexus_chat_agent, _current_thread_id
    from seer.agents.nexus.cost_callback import CostCapCallbackHandler, set_chat_runtime_context, clear_chat_runtime_context
    from seer.api.agents.checkpointer import _recreate_checkpointer  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.api.agents.workflow.chat_services import (
        ChatOrchestrator, CheckpointerHealthService, IncompleteToolCallDetector, IncompleteToolCallRecoveryService,
    )
    from seer.api.agents.workflow.streaming import (
        stream_agent_tokens, extract_post_stream_state, save_stream_result, build_final_sse_event, mark_session_failed,
    )
    from seer.utilities.langfuse_tracing import merge_nexus_langfuse_callbacks
    from seer.worker.tasks.chat import _get_user_settings_and_context, _handle_agent_result
    from seer.observability.exceptions import RunCostCapExceeded

    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)
    model = chat_request.model or config.default_llm_model

    session, thread_id, session_id = await SessionService.get_or_create_session(
        workflow=workflow, user=user,
        thread_id=chat_request.thread_id, session_id=chat_request.session_id,
    )

    if session.current_execution_status in [ChatExecutionStatus.QUEUED, ChatExecutionStatus.RUNNING]:
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Execution already in progress",
                      detail="Cannot start new execution while another is in progress", status=409)

    session.current_workflow_state = await workflow_state_snapshot(workflow)
    await session.save(update_fields=['current_workflow_state'])
    await save_chat_message(session_id=session_id, role="user", content=chat_request.message)

    async def event_generator():
        session.current_execution_status = ChatExecutionStatus.RUNNING
        session.current_execution_started_at = datetime.now(timezone.utc)
        session.current_execution_error = None
        session.pending_interrupt_type = None
        session.pending_interrupt_data = None
        await session.save(update_fields=[
            'current_execution_status', 'current_execution_started_at',
            'current_execution_error', 'pending_interrupt_type', 'pending_interrupt_data',
        ])

        try:
            checkpointer = await get_checkpointer()
            agent = await create_nexus_chat_agent(
                model=model, checkpointer=checkpointer,
                user_id=user.user_id, current_query=chat_request.message, workflow_id=workflow.workflow_id,
            )
            max_agent_steps, runtime_context = await _get_user_settings_and_context(user, thread_id)
            set_chat_runtime_context(runtime_context)

            orchestrator = ChatOrchestrator(
                agent=agent, checkpointer=checkpointer,
                health_service=CheckpointerHealthService(), detector=IncompleteToolCallDetector(),
                recovery_service=IncompleteToolCallRecoveryService(), reconnect_func=_recreate_checkpointer,
            )
            cost_callback = CostCapCallbackHandler()
            config_dict = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_agent_steps, "callbacks": [cost_callback]}

            # Handle incomplete tool call recovery
            if await orchestrator._check_for_incomplete_tool_calls(config_dict):  # pylint: disable=protected-access # Reason: Internal module helper
                async for sse_line in _handle_recovery_stream(orchestrator, chat_request.message, config_dict, thread_id, session, session_id):
                    yield sse_line
                return

            # Normal streaming path
            config_with_langfuse = merge_nexus_langfuse_callbacks(config_dict)
            token_ctx = _current_thread_id.set(thread_id)
            try:
                full_response = ""
                input_msgs = {"messages": [HumanMessage(content=chat_request.message)]}
                async for token in stream_agent_tokens(agent, input_msgs, config_with_langfuse, user.user_id):
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

                state = await extract_post_stream_state(agent, config_with_langfuse)
                if not full_response:
                    full_response = state["full_response"]
                proposal = await save_stream_result(
                    session, session_id, thread_id, full_response,
                    state["all_thinking"], state["interrupt_required"], state["interrupt_data"],
                )
                yield build_final_sse_event(
                    session_id, thread_id, state["all_thinking"],
                    state["interrupt_required"], state["interrupt_data"], proposal,
                )
            finally:
                _current_thread_id.reset(token_ctx)
                clear_chat_runtime_context()

        except RunCostCapExceeded as e:
            logger.warning("Chat cost cap exceeded during stream", extra={"session_id": session_id})
            await mark_session_failed(session, "cost_cap_exceeded", str(e.to_dict()), 402)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: SSE must send error event
            logger.error("Streaming chat failed", exc_info=True, extra={"session_id": session_id})
            await mark_session_failed(session, "execution_error", str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{workflow_id}/chat/resume/stream")
async def stream_resume_chat_endpoint(  # pylint: disable=too-many-locals # Reason: SSE endpoint with nested generator
    request: Request,
    workflow_id: str,
    resume_data: ChatResumeRequest,
):
    """Stream resumed chat response via SSE after interrupt."""
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.database.workflow_models import ChatExecutionStatus
    from seer.agents.nexus import create_nexus_chat_agent, _current_thread_id
    from seer.agents.nexus.cost_callback import CostCapCallbackHandler, set_chat_runtime_context, clear_chat_runtime_context
    from seer.api.agents.workflow.streaming import (
        stream_agent_tokens, extract_post_stream_state, save_stream_result, build_final_sse_event, mark_session_failed,
    )
    from seer.utilities.langfuse_tracing import merge_nexus_langfuse_callbacks
    from seer.worker.tasks.chat import _get_user_settings_and_context
    from seer.observability.exceptions import RunCostCapExceeded

    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)

    checkpointer = await get_checkpointer()
    session = await get_chat_session_by_thread_id(resume_data.thread_id, workflow)
    if not session:
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Session not found",
                      detail=f"Chat session not found for thread_id: {resume_data.thread_id}", status=404)
    assert session is not None
    session_id = session.id
    thread_id = resume_data.thread_id

    resume_command = await _build_resume_command(resume_data, checkpointer, session_id)

    async def event_generator():
        session.current_execution_status = ChatExecutionStatus.RUNNING
        session.current_execution_started_at = datetime.now(timezone.utc)
        session.current_execution_error = None
        session.pending_interrupt_type = None
        session.pending_interrupt_data = None
        await session.save(update_fields=[
            'current_execution_status', 'current_execution_started_at',
            'current_execution_error', 'pending_interrupt_type', 'pending_interrupt_data',
        ])

        try:
            agent = await create_nexus_chat_agent(
                model=config.default_llm_model, checkpointer=checkpointer,
                user_id=user.user_id, workflow_id=workflow.workflow_id,
            )
            max_agent_steps, runtime_context = await _get_user_settings_and_context(user, thread_id)
            set_chat_runtime_context(runtime_context)

            cost_callback = CostCapCallbackHandler()
            config_dict = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_agent_steps, "callbacks": [cost_callback]}
            config_with_langfuse = merge_nexus_langfuse_callbacks(config_dict)

            token_ctx = _current_thread_id.set(thread_id)
            try:
                full_response = ""
                async for token in stream_agent_tokens(agent, resume_command, config_with_langfuse, user.user_id):
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

                state = await extract_post_stream_state(agent, config_with_langfuse)
                if not full_response:
                    full_response = state["full_response"]
                proposal = await save_stream_result(
                    session, session_id, thread_id, full_response,
                    state["all_thinking"], state["interrupt_required"], state["interrupt_data"],
                )
                yield build_final_sse_event(
                    session_id, thread_id, state["all_thinking"],
                    state["interrupt_required"], state["interrupt_data"], proposal,
                )
            finally:
                _current_thread_id.reset(token_ctx)
                clear_chat_runtime_context()

        except RunCostCapExceeded as e:
            logger.warning("Chat cost cap exceeded during resume stream", extra={"session_id": session_id})
            await mark_session_failed(session, "cost_cap_exceeded", str(e.to_dict()), 402)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: SSE must send error event
            logger.error("Streaming resume failed", exc_info=True, extra={"session_id": session_id})
            await mark_session_failed(session, "execution_error", str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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

    Execution runs in background. Client must poll /chat/status/{session_id} for results.
    This makes the API resilient to client disconnections.
    """
    from seer.database.workflow_models import ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: Only needed for status handling
    from seer.worker.tasks.chat import chat_execution_task  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    logger.info("Chat request received: workflow_id=%s, message_length=%d", workflow_id, len(chat_request.message))
    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)

    # LLM credit limit check enforced by UsageLimitMiddleware
    model = chat_request.model or config.default_llm_model

    # Get or create session
    session, thread_id, session_id = await SessionService.get_or_create_session(
        workflow=workflow,
        user=user,
        thread_id=chat_request.thread_id,
        session_id=chat_request.session_id,
    )

    # Check if session already has execution in progress
    if session.current_execution_status in [ChatExecutionStatus.QUEUED, ChatExecutionStatus.RUNNING]:
        raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Execution already in progress",
            detail="Cannot start new execution while another is in progress",
            status=409
        )

    # Save current workflow state to session for UI/persistence
    session.current_workflow_state = await workflow_state_snapshot(workflow)
    await session.save(update_fields=['current_workflow_state'])

    # Save user message first (before async execution)
    await save_chat_message(
        session_id=session_id,
        role="user",
        content=chat_request.message,
    )


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

    return ChatResponse(
        response="",
        session_id=session_id,
        thread_id=thread_id,
        execution_status=ChatExecutionStatus.QUEUED.value,
        execution_task_id=task.task_id,
    )


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
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)
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
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)

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
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)
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
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)
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
async def resume_chat_endpoint(
    request: Request,
    workflow_id: str,
    resume_data: ChatResumeRequest,
) -> ChatResponse:
    """
    Resume chat after interrupt (clarification question or other interrupt type).

    For clarification questions, provide 'answer' with selected values.
    For other interrupts, provide 'command' with raw Command data.

    Execution runs in background. Client must poll /chat/status/{session_id} for results.
    """
    from seer.worker.tasks.chat import chat_resume_task  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.database.workflow_models import ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: Only needed for status handling

    logger.info("Resume request received: workflow_id=%s, thread_id=%s", workflow_id, resume_data.thread_id)
    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)

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

    # Build resume command (validates answer)
    resume_command = await _build_resume_command(resume_data, checkpointer, session_id)

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

    return ChatResponse(
        response="",
        session_id=session_id,
        thread_id=resume_data.thread_id,
        execution_status=ChatExecutionStatus.QUEUED.value,
        execution_task_id=task.task_id,
    )


@router.get("/{workflow_id}/proposals/{proposal_id}", response_model=WorkflowProposalPublic)
async def get_proposal_endpoint(
    request: Request,
    workflow_id: str,
    proposal_id: int,
) -> WorkflowProposalPublic:
    """Fetch a single workflow proposal."""
    user = _require_user(request)
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)
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
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)
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
    org, membership = _get_org_context(request)
    workflow = await get_workflow(user, workflow_id, organization=org, membership=membership)
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
