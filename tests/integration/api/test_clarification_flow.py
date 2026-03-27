"""
Integration tests for clarification question flow.

Tests session state validation, answer validation, and interrupt handling
with REAL database records. Only external infrastructure (Redis, task queue,
Postgres checkpointer) is mocked — no business logic mocking.

Replaces mock-heavy test that had 120 mock sites testing against air.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from seer.database.workflow_models import (
    ChatExecutionStatus,
    WorkflowChatSession,
)


# =============================================================================
# Shared interrupt payloads
# =============================================================================

SINGLE_CHOICE_INTERRUPT = {
    "type": "clarification_questions",
    "questions": [
        {
            "question_id": "q_email",
            "question": "Which email provider should we use?",
            "question_type": "single_choice",
            "options": [
                {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                {"value": "outlook", "label": "Outlook", "is_wildcard": False},
                {"value": "other", "label": "Other", "is_wildcard": True},
            ],
            "min_selections": 1,
            "max_selections": None,
            "reasoning": "Found multiple email providers",
        }
    ],
}

MULTI_CHOICE_INTERRUPT = {
    "type": "clarification_questions",
    "questions": [
        {
            "question_id": "q_multi",
            "question": "Which integrations to enable?",
            "question_type": "multi_choice",
            "options": [
                {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                {"value": "slack", "label": "Slack", "is_wildcard": False},
                {"value": "github", "label": "GitHub", "is_wildcard": False},
            ],
            "min_selections": 1,
            "max_selections": 3,
            "reasoning": "User wants multiple integrations",
        }
    ],
}

BATCH_INTERRUPT = {
    "type": "clarification_questions",
    "questions": [
        {
            "question_id": "q_email",
            "question": "Which email provider?",
            "question_type": "single_choice",
            "options": [
                {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                {"value": "outlook", "label": "Outlook", "is_wildcard": False},
            ],
            "min_selections": 1,
            "max_selections": None,
            "reasoning": "Need email config",
        },
        {
            "question_id": "q_notify",
            "question": "Which notification channels?",
            "question_type": "multi_choice",
            "options": [
                {"value": "slack", "label": "Slack", "is_wildcard": False},
                {"value": "email", "label": "Email", "is_wildcard": False},
            ],
            "min_selections": 1,
            "max_selections": 2,
            "reasoning": "Need notification config",
        },
    ],
}


def _mock_checkpointer_with_interrupt(interrupt_data):
    """Return a mock checkpointer whose aget_tuple returns the given interrupt."""
    checkpointer = AsyncMock()
    checkpointer.aget_tuple.return_value = MagicMock(
        checkpoint={
            "channel_values": {
                "__interrupt__": [interrupt_data]
            }
        }
    )
    return checkpointer


async def _empty_sse(*args, **kwargs):
    """Stub async generator for SSE — yields nothing."""
    return
    yield  # noqa: unreachable — makes this an async generator


# =============================================================================
# Session State Guards (Real DB)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestClarificationSessionGuards:
    """Router rejects invalid state transitions with real DB sessions.

    Only external infrastructure is mocked (task queue, Redis streaming).
    Session state validation uses real DB records.
    """

    @pytest.fixture
    async def client(self, mock_app: FastAPI, db_engine, test_user, test_workflow):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
        """Create authenticated API client with workflow router."""
        from seer.api.agents.workflow import router  # pylint: disable=import-outside-toplevel

        mock_app.include_router(router)

        async def auth_middleware(request: Request, call_next):
            request.state.db_user = test_user
            return await call_next(request)

        mock_app.middleware("http")(auth_middleware)

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c, test_workflow

    async def test_rejects_fresh_chat_for_interrupted_session(self, client):
        """POST /chat must return 409 when session is already interrupted."""
        api_client, workflow = client

        session = await WorkflowChatSession.create(
            workflow=workflow,
            user=workflow.user,
            thread_id="interrupted-thread",
            title="Interrupted Session",
            current_execution_status=ChatExecutionStatus.INTERRUPTED,
            pending_interrupt_type="clarification_questions",
            pending_interrupt_data=SINGLE_CHOICE_INTERRUPT,
        )

        response = await api_client.post(
            f"/nexus/{workflow.workflow_id}/chat",
            json={"message": "Try again", "session_id": session.id},
        )

        assert response.status_code == 409
        assert "/chat/resume" in response.text

    async def test_rejects_resume_without_pending_interrupt(self, client):
        """POST /chat/resume must return 409 when session is already completed."""
        api_client, workflow = client

        await WorkflowChatSession.create(
            workflow=workflow,
            user=workflow.user,
            thread_id="completed-thread",
            title="Completed Session",
            current_execution_status=ChatExecutionStatus.COMPLETED,
            pending_interrupt_type=None,
            pending_interrupt_data=None,
        )

        with patch(
            "seer.api.agents.workflow.router.get_checkpointer",
            return_value=AsyncMock(),
        ):
            response = await api_client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": "completed-thread",
                    "answers": {
                        "answers": [
                            {"question_id": "q_email", "selected_values": ["gmail"]}
                        ]
                    },
                },
            )

        assert response.status_code == 409
        assert "Start a new /chat request" in response.text

    async def test_rejects_resume_with_multiple_payload_types(self, client):
        """Resume payload must contain exactly one of answers, message, or command."""
        api_client, workflow = client

        response = await api_client.post(
            f"/nexus/{workflow.workflow_id}/chat/resume",
            json={
                "thread_id": "any-thread",
                "message": "Use Gmail",
                "answers": {
                    "answers": [
                        {"question_id": "q_email", "selected_values": ["gmail"]}
                    ]
                },
            },
        )

        assert response.status_code == 422
        assert "Exactly one of answers, message, or command must be provided" in response.text


# =============================================================================
# Answer Validation (Real DB + checkpointer mock for question data)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestClarificationAnswerValidation:
    """Router validates clarification answers against the original question schema.

    Real DB for session state. Checkpointer mock only returns the stored
    question data so the validation logic can compare answers against it.
    """

    @pytest.fixture
    async def client(self, mock_app: FastAPI, db_engine, test_user, test_workflow):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
        """Create authenticated API client with workflow router."""
        from seer.api.agents.workflow import router  # pylint: disable=import-outside-toplevel

        mock_app.include_router(router)

        async def auth_middleware(request: Request, call_next):
            request.state.db_user = test_user
            return await call_next(request)

        mock_app.middleware("http")(auth_middleware)

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c, test_workflow

    async def _create_interrupted_session(self, workflow, interrupt_data, thread_id="val-thread"):
        """Helper: create an interrupted session with the given interrupt data."""
        return await WorkflowChatSession.create(
            workflow=workflow,
            user=workflow.user,
            thread_id=thread_id,
            title="Validation Session",
            current_execution_status=ChatExecutionStatus.INTERRUPTED,
            pending_interrupt_type="clarification_questions",
            pending_interrupt_data=interrupt_data,
        )

    async def test_rejects_invalid_selection(self, client):
        """Selected value not in available options returns 400."""
        api_client, workflow = client
        await self._create_interrupted_session(workflow, SINGLE_CHOICE_INTERRUPT, "invalid-sel-thread")

        checkpointer = _mock_checkpointer_with_interrupt(SINGLE_CHOICE_INTERRUPT)

        with patch(
            "seer.api.agents.workflow.router.get_checkpointer",
            return_value=checkpointer,
        ):
            response = await api_client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": "invalid-sel-thread",
                    "answers": {
                        "answers": [
                            {
                                "question_id": "q_email",
                                "selected_values": ["invalid_provider"],
                                "custom_input": None,
                            }
                        ]
                    },
                },
            )

        assert response.status_code == 400
        data = response.json()
        assert (
            data.get("title") == "Invalid selections"
            or "Invalid selections" in str(data.get("detail", {}))
        )

    async def test_wildcard_requires_custom_input(self, client):
        """Selecting a wildcard option without custom_input returns 400."""
        api_client, workflow = client
        await self._create_interrupted_session(workflow, SINGLE_CHOICE_INTERRUPT, "wildcard-thread")

        checkpointer = _mock_checkpointer_with_interrupt(SINGLE_CHOICE_INTERRUPT)

        with patch(
            "seer.api.agents.workflow.router.get_checkpointer",
            return_value=checkpointer,
        ):
            response = await api_client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": "wildcard-thread",
                    "answers": {
                        "answers": [
                            {
                                "question_id": "q_email",
                                "selected_values": ["other"],
                                "custom_input": None,
                            }
                        ]
                    },
                },
            )

        assert response.status_code == 400
        data = response.json()
        assert (
            data.get("title") == "Custom input required"
            or "custom input" in str(data.get("detail", {})).lower()
        )

    async def test_wildcard_with_custom_input_succeeds(self, client):
        """Wildcard selection with custom_input passes validation and enqueues resume."""
        api_client, workflow = client
        await self._create_interrupted_session(workflow, SINGLE_CHOICE_INTERRUPT, "wildcard-ok-thread")

        checkpointer = _mock_checkpointer_with_interrupt(SINGLE_CHOICE_INTERRUPT)
        mock_task_result = MagicMock()
        mock_task_result.task_id = "mock-task-id"

        with (
            patch("seer.api.agents.workflow.router.get_checkpointer", return_value=checkpointer),
            patch("seer.worker.tasks.chat.chat_resume_task") as mock_resume_task,
            patch("seer.api.agents.workflow.router.get_stream_watermark", new_callable=AsyncMock, return_value="0"),
            patch("seer.agents.nexus.stream_publisher.StreamPublisher") as mock_publisher_cls,
            patch("seer.api.agents.workflow.router.stream_events_sse", new=_empty_sse),
        ):
            mock_publisher_cls.return_value = AsyncMock()
            mock_resume_task.kiq = AsyncMock(return_value=mock_task_result)

            response = await api_client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": "wildcard-ok-thread",
                    "answers": {
                        "answers": [
                            {
                                "question_id": "q_email",
                                "selected_values": ["other"],
                                "custom_input": "ProtonMail",
                            }
                        ]
                    },
                },
            )

        assert response.status_code == 200

    async def test_valid_answer_enqueues_resume(self, client):
        """Valid answer passes validation, enqueues resume task, and updates session to QUEUED."""
        api_client, workflow = client
        session = await self._create_interrupted_session(
            workflow, SINGLE_CHOICE_INTERRUPT, "valid-ans-thread"
        )

        checkpointer = _mock_checkpointer_with_interrupt(SINGLE_CHOICE_INTERRUPT)
        mock_task_result = MagicMock()
        mock_task_result.task_id = "mock-task-id"
        captured_args = {}

        async def capture_kiq(**kwargs):
            captured_args.update(kwargs)
            return mock_task_result

        with (
            patch("seer.api.agents.workflow.router.get_checkpointer", return_value=checkpointer),
            patch("seer.worker.tasks.chat.chat_resume_task") as mock_resume_task,
            patch("seer.api.agents.workflow.router.get_stream_watermark", new_callable=AsyncMock, return_value="0"),
            patch("seer.agents.nexus.stream_publisher.StreamPublisher") as mock_publisher_cls,
            patch("seer.api.agents.workflow.router.stream_events_sse", new=_empty_sse),
        ):
            mock_publisher_cls.return_value = AsyncMock()
            mock_resume_task.kiq = AsyncMock(side_effect=capture_kiq)

            response = await api_client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": "valid-ans-thread",
                    "answers": {
                        "answers": [
                            {
                                "question_id": "q_email",
                                "selected_values": ["gmail"],
                                "custom_input": None,
                            }
                        ]
                    },
                },
            )

        assert response.status_code == 200

        # Verify resume task enqueued with correct data
        assert captured_args["thread_id"] == "valid-ans-thread"
        assert captured_args["resume_command_data"] == [
            {"question_id": "q_email", "selected_values": ["gmail"], "custom_input": None}
        ]

        # Session should be QUEUED (ready for task execution)
        await session.refresh_from_db()
        assert session.current_execution_status == ChatExecutionStatus.QUEUED
        assert session.pending_interrupt_type is None
        assert session.pending_interrupt_data is None

    async def test_free_text_resume_builds_correct_command(self, client):
        """Free-text reply during interrupt builds correct resume command data."""
        api_client, workflow = client
        await self._create_interrupted_session(
            workflow, SINGLE_CHOICE_INTERRUPT, "free-text-thread"
        )

        mock_task_result = MagicMock()
        mock_task_result.task_id = "mock-task-id"
        captured_args = {}

        async def capture_kiq(**kwargs):
            captured_args.update(kwargs)
            return mock_task_result

        with (
            patch("seer.api.agents.workflow.router.get_checkpointer", return_value=AsyncMock()),
            patch("seer.worker.tasks.chat.chat_resume_task") as mock_resume_task,
            patch("seer.api.agents.workflow.router.get_stream_watermark", new_callable=AsyncMock, return_value="0"),
            patch("seer.agents.nexus.stream_publisher.StreamPublisher") as mock_publisher_cls,
            patch("seer.api.agents.workflow.router.stream_events_sse", new=_empty_sse),
        ):
            mock_publisher_cls.return_value = AsyncMock()
            mock_resume_task.kiq = AsyncMock(side_effect=capture_kiq)

            response = await api_client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": "free-text-thread",
                    "message": "Use Gmail for invoices from finance@example.com.",
                },
            )

        assert response.status_code == 200
        assert captured_args["resume_command_data"] == {
            "type": "free_text_response",
            "message": "Use Gmail for invoices from finance@example.com.",
            "source": "chat_resume_message",
        }

    async def test_batch_answers_all_validated(self, client):
        """Multiple questions require all answers present and valid."""
        api_client, workflow = client
        await self._create_interrupted_session(
            workflow, BATCH_INTERRUPT, "batch-ans-thread"
        )

        checkpointer = _mock_checkpointer_with_interrupt(BATCH_INTERRUPT)
        mock_task_result = MagicMock()
        mock_task_result.task_id = "mock-task-id"
        captured_args = {}

        async def capture_kiq(**kwargs):
            captured_args.update(kwargs)
            return mock_task_result

        with (
            patch("seer.api.agents.workflow.router.get_checkpointer", return_value=checkpointer),
            patch("seer.worker.tasks.chat.chat_resume_task") as mock_resume_task,
            patch("seer.api.agents.workflow.router.get_stream_watermark", new_callable=AsyncMock, return_value="0"),
            patch("seer.agents.nexus.stream_publisher.StreamPublisher") as mock_publisher_cls,
            patch("seer.api.agents.workflow.router.stream_events_sse", new=_empty_sse),
        ):
            mock_publisher_cls.return_value = AsyncMock()
            mock_resume_task.kiq = AsyncMock(side_effect=capture_kiq)

            response = await api_client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": "batch-ans-thread",
                    "answers": {
                        "answers": [
                            {"question_id": "q_email", "selected_values": ["gmail"], "custom_input": None},
                            {"question_id": "q_notify", "selected_values": ["slack"], "custom_input": None},
                        ]
                    },
                },
            )

        assert response.status_code == 200
        assert len(captured_args["resume_command_data"]) == 2
        assert captured_args["resume_command_data"][0]["question_id"] == "q_email"
        assert captured_args["resume_command_data"][1]["question_id"] == "q_notify"


# =============================================================================
# Interrupt Handling — _handle_agent_result with Real DB
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestClarificationInterruptHandling:
    """Tests interrupt detection and session state updates with real DB.

    Calls _handle_agent_result directly — this is the function that
    inspects the agent's return value and updates the session in the database.
    No router, no HTTP client, no Redis — just real DB and real logic.
    """

    async def _create_running_session(self, workflow, thread_id="int-thread"):
        """Create a session in RUNNING state (as it would be during task execution)."""
        return await WorkflowChatSession.create(
            workflow=workflow,
            user=workflow.user,
            thread_id=thread_id,
            title="Running Session",
            current_execution_status=ChatExecutionStatus.RUNNING,
            current_execution_task_id="task-123",
        )

    async def test_interrupt_detected_sets_session_interrupted(self, db_engine, test_workflow):
        """Agent result with __interrupt__ sets session to INTERRUPTED with correct data."""
        from seer.worker.tasks.chat import _handle_agent_result  # pylint: disable=import-outside-toplevel

        session = await self._create_running_session(test_workflow)

        # Simulate agent result with interrupt — this is what LangGraph returns
        # when the agent calls interrupt()
        result = {
            "messages": [MagicMock(content="I need to know which email provider to use.")],
            "__interrupt__": [SINGLE_CHOICE_INTERRUPT],
        }

        await _handle_agent_result(session, result, session.thread_id, session.id)

        # Verify real DB state
        await session.refresh_from_db()
        assert session.current_execution_status == ChatExecutionStatus.INTERRUPTED
        assert session.pending_interrupt_type == "clarification_questions"
        assert session.pending_interrupt_data is not None
        assert session.current_execution_task_id is None

        # Verify stored question data matches
        questions = session.pending_interrupt_data.get("questions", [])
        assert len(questions) == 1
        assert questions[0]["question_id"] == "q_email"
        assert questions[0]["question_type"] == "single_choice"
        assert len(questions[0]["options"]) == 3

    async def test_multi_choice_interrupt_stored_correctly(self, db_engine, test_workflow):
        """Multi-choice interrupt stores the correct question type and options."""
        from seer.worker.tasks.chat import _handle_agent_result  # pylint: disable=import-outside-toplevel

        session = await self._create_running_session(test_workflow, "multi-int-thread")

        result = {
            "messages": [MagicMock(content="Which integrations?")],
            "__interrupt__": [MULTI_CHOICE_INTERRUPT],
        }

        await _handle_agent_result(session, result, session.thread_id, session.id)

        await session.refresh_from_db()
        assert session.current_execution_status == ChatExecutionStatus.INTERRUPTED
        questions = session.pending_interrupt_data["questions"]
        assert questions[0]["question_type"] == "multi_choice"
        assert len(questions[0]["options"]) == 3

    async def test_batch_interrupt_stores_multiple_questions(self, db_engine, test_workflow):
        """Batch interrupt with multiple questions stores all of them."""
        from seer.worker.tasks.chat import _handle_agent_result  # pylint: disable=import-outside-toplevel

        session = await self._create_running_session(test_workflow, "batch-int-thread")

        result = {
            "messages": [MagicMock(content="I have a couple of questions.")],
            "__interrupt__": [BATCH_INTERRUPT],
        }

        await _handle_agent_result(session, result, session.thread_id, session.id)

        await session.refresh_from_db()
        assert session.current_execution_status == ChatExecutionStatus.INTERRUPTED
        questions = session.pending_interrupt_data["questions"]
        assert len(questions) == 2
        assert questions[0]["question_id"] == "q_email"
        assert questions[1]["question_id"] == "q_notify"

    async def test_no_interrupt_completes_session(self, db_engine, test_workflow):
        """Agent result without __interrupt__ sets session to COMPLETED."""
        from seer.worker.tasks.chat import _handle_agent_result  # pylint: disable=import-outside-toplevel

        session = await self._create_running_session(test_workflow, "complete-thread")

        result = {
            "messages": [MagicMock(content="Done! I've set up the email integration.")],
        }

        with patch(
            "seer.worker.tasks.chat.extract_session_memories",
            new=MagicMock(kiq=AsyncMock()),
        ):
            await _handle_agent_result(session, result, session.thread_id, session.id)

        await session.refresh_from_db()
        assert session.current_execution_status == ChatExecutionStatus.COMPLETED
        assert session.pending_interrupt_type is None
        assert session.pending_interrupt_data is None
        assert session.current_execution_task_id is None


# =============================================================================
# InterruptHandler Unit Logic (Pure — No DB, No Mocks)
# =============================================================================


@pytest.mark.integration
class TestInterruptHandlerExtraction:
    """Tests InterruptHandler.extract_interrupt_from_result — pure logic."""

    def _extract(self, result):
        from seer.api.agents.workflow.chat_services import InterruptHandler  # pylint: disable=import-outside-toplevel
        return InterruptHandler.extract_interrupt_from_result(result)

    def test_dict_interrupt_detected(self):
        """__interrupt__ with dict list is detected correctly."""
        result = {"messages": [], "__interrupt__": [SINGLE_CHOICE_INTERRUPT]}
        found, data = self._extract(result)
        assert found is True
        assert data["type"] == "clarification_questions"

    def test_no_interrupt_returns_false(self):
        """Result without __interrupt__ returns (False, None)."""
        result = {"messages": [MagicMock(content="done")]}
        found, data = self._extract(result)
        assert found is False
        assert data is None

    def test_empty_interrupt_list_still_detected(self):
        """Empty __interrupt__ list is still treated as truthy by the handler.

        This is the actual behavior — the handler stringifies the empty list.
        A production fix could make this return (False, None) instead.
        """
        result = {"messages": [], "__interrupt__": []}
        found, data = self._extract(result)
        # Current behavior: empty list still triggers interrupt detection
        # because the code falls through to the `else` branch
        assert found is True
        assert data == {"value": "[]"}

    def test_interrupt_with_value_attribute(self):
        """LangGraph Interrupt objects with .value attribute are unwrapped."""
        mock_interrupt = MagicMock()
        mock_interrupt.value = SINGLE_CHOICE_INTERRUPT
        result = {"messages": [], "__interrupt__": [mock_interrupt]}
        found, data = self._extract(result)
        assert found is True
        assert data["type"] == "clarification_questions"

    def test_non_dict_result_returns_false(self):
        """Non-dict result returns (False, None)."""
        found, data = self._extract("not a dict")
        assert found is False
        assert data is None
