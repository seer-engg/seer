"""
Integration tests for clarification question flow.

Tests the full flow: chat -> interrupt -> resume with clarification questions.

NOTE: These tests use async_mode=True (the default) and mock taskiq tasks.
The tests simulate background task execution by calling task functions directly
after the API enqueues them.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


async def _empty_sse(*args, **kwargs):
    """Async generator stub that yields nothing — replaces stream_events_sse in tests."""
    return
    yield  # noqa: unreachable — makes this an async generator


@pytest.mark.integration
@pytest.mark.asyncio
class TestClarificationFlow:
    """Integration tests for clarification question flow."""

    @pytest.fixture
    async def workflow_client(self, mock_app: FastAPI, db_engine, test_user, test_workflow):  # pylint: disable=unused-argument # Reason: db_engine needed for database initialization
        """Create authenticated API client with workflow router."""
        from seer.api.agents.workflow import router  # pylint: disable=import-outside-toplevel  # Dynamic import for test fixture

        # Add router to test app
        mock_app.include_router(router)

        # Mock authentication to inject test_user
        from fastapi import Request  # pylint: disable=import-outside-toplevel  # Dynamic import for test fixture
        async def mock_auth_middleware(request: Request, call_next):
            request.state.db_user = test_user
            response = await call_next(request)
            return response

        mock_app.middleware("http")(mock_auth_middleware)

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, test_workflow

    @pytest.fixture
    def mock_task_result(self):
        """Create a mock taskiq task result."""
        result = MagicMock()
        result.task_id = "mock-task-id-123"
        return result

    async def test_interrupt_triggered_by_clarification_tool(
        self,
        workflow_client,
        mock_task_result
    ):
        """Test that clarification questions trigger interrupt via async task."""
        from seer.database.workflow_models import WorkflowChatSession, ChatExecutionStatus  # pylint: disable=import-outside-toplevel

        client, workflow = workflow_client

        # Use batch format (clarification_questions) instead of single question
        interrupt_data = {
            "type": "clarification_questions",
            "questions": [
                {
                    "question_id": "q_test123",
                    "question": "Which email provider should we use?",
                    "question_type": "single_choice",
                    "options": [
                        {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                        {"value": "outlook", "label": "Outlook", "is_wildcard": False},
                        {"value": "other", "label": "Other", "is_wildcard": True},
                    ],
                    "min_selections": 1,
                    "max_selections": None,
                    "reasoning": "Found multiple email providers"
                }
            ]
        }

        # Mock agent to return interrupt (used by the task)
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="I need to know which email provider to use.")
            ],
            "__interrupt__": [interrupt_data]
        }

        # Capture task arguments when .kiq() is called
        captured_task_args = {}

        async def capture_kiq(**kwargs):
            captured_task_args.update(kwargs)
            return mock_task_result

        with patch('seer.api.agents.workflow.router.get_checkpointer') as mock_get_checkpointer, \
             patch('seer.worker.tasks.chat.chat_execution_task') as mock_chat_task, \
             patch('seer.api.agents.workflow.router.get_stream_watermark', new_callable=AsyncMock, return_value="0"), \
             patch('seer.agents.nexus.stream_publisher.StreamPublisher') as mock_publisher_cls, \
             patch('seer.api.agents.workflow.router.stream_events_sse', new=_empty_sse):

            mock_publisher_cls.return_value = AsyncMock()
            mock_get_checkpointer.return_value = AsyncMock()
            mock_chat_task.kiq = AsyncMock(side_effect=capture_kiq)

            # Send chat message (async_mode=True by default)
            response = await client.post(
                f"/nexus/{workflow.workflow_id}/chat",
                json={
                    "message": "Set up email integration"
                }
            )

            assert response.status_code == 200
            # session_id comes from captured kiq args — response is SSE, not JSON
            session_id = captured_task_args["session_id"]

        # Verify task was enqueued with correct args
        assert "session_id" in captured_task_args
        assert captured_task_args["message"] == "Set up email integration"

        # Now simulate the background task execution by patching what it uses
        # and directly updating the session state (simulating task completion)
        with patch('seer.worker.tasks.chat.create_nexus_chat_agent', return_value=mock_agent), \
             patch('seer.worker.tasks.chat.get_checkpointer') as mock_task_checkpointer:

            mock_task_checkpointer.return_value = AsyncMock()

            # Import and run the task directly to simulate background execution
            from seer.worker.tasks.chat import chat_execution_task
            await chat_execution_task(
                session_id=session_id,
                user_id=workflow.user_id,
                message="Set up email integration",
                workflow_id=workflow.id,
            )

        # Now check the session state - should have interrupt
        session = await WorkflowChatSession.get(id=session_id)
        assert session.current_execution_status == ChatExecutionStatus.INTERRUPTED
        assert session.pending_interrupt_type == "clarification_questions"
        assert session.pending_interrupt_data is not None

        # Verify structured questions in interrupt_data
        questions = session.pending_interrupt_data.get("questions", [])
        assert len(questions) == 1
        assert questions[0].get("question_id") == "q_test123"

    async def test_resume_with_valid_answer(
        self,
        workflow_client,
        mock_task_result
    ):
        """Test resuming with valid clarification answers via async task."""
        from seer.database.workflow_models import WorkflowChatSession, ChatExecutionStatus  # pylint: disable=import-outside-toplevel  # Dynamic import for test

        client, workflow = workflow_client

        # Create chat session in database for the thread
        thread_id = "test-thread-123"
        session = await WorkflowChatSession.create(
            workflow=workflow,
            user=workflow.user,
            thread_id=thread_id,
            title="Test Session"
        )

        # Mock checkpointer to return stored interrupt (used for validation)
        mock_checkpointer = AsyncMock()
        mock_checkpointer.aget_tuple.return_value = MagicMock(
            checkpoint={
                "channel_values": {
                    "__interrupt__": [
                        {
                            "type": "clarification_questions",
                            "questions": [
                                {
                                    "question_id": "q_test123",
                                    "options": [
                                        {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                                        {"value": "outlook", "label": "Outlook", "is_wildcard": False},
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        )

        # Mock agent to return continued response (used by the task)
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="Great! I'll set up Gmail integration.")
            ]
        }

        # Capture task arguments
        captured_task_args = {}

        async def capture_kiq(**kwargs):
            captured_task_args.update(kwargs)
            return mock_task_result

        with patch('seer.api.agents.workflow.router.get_checkpointer', return_value=mock_checkpointer), \
             patch('seer.worker.tasks.chat.chat_resume_task') as mock_resume_task, \
             patch('seer.api.agents.workflow.router.get_stream_watermark', new_callable=AsyncMock, return_value="0"), \
             patch('seer.agents.nexus.stream_publisher.StreamPublisher') as mock_publisher_cls, \
             patch('seer.api.agents.workflow.router.stream_events_sse', new=_empty_sse):

            mock_publisher_cls.return_value = AsyncMock()
            mock_resume_task.kiq = AsyncMock(side_effect=capture_kiq)

            # Resume with answers (batch format)
            response = await client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": thread_id,
                    "answers": {
                        "answers": [
                            {
                                "question_id": "q_test123",
                                "selected_values": ["gmail"],
                                "custom_input": None
                            }
                        ]
                    }
                }
            )

            assert response.status_code == 200

        # Verify task was enqueued
        assert "session_id" in captured_task_args
        assert captured_task_args["thread_id"] == thread_id

        # Simulate background task execution
        with patch('seer.worker.tasks.chat.create_nexus_chat_agent', return_value=mock_agent), \
             patch('seer.worker.tasks.chat.get_checkpointer', return_value=mock_checkpointer):

            from seer.worker.tasks.chat import chat_resume_task
            await chat_resume_task(
                session_id=session.id,
                user_id=workflow.user_id,
                thread_id=thread_id,
                resume_command_data=captured_task_args["resume_command_data"],
                workflow_id=workflow.id,
            )

        # Verify session completed (no interrupt)
        await session.refresh_from_db()
        assert session.current_execution_status == ChatExecutionStatus.COMPLETED

    async def test_resume_with_invalid_selection(
        self,
        workflow_client
    ):
        """Test that invalid selections are rejected (validation happens before async task)."""
        from seer.database.workflow_models import WorkflowChatSession  # pylint: disable=import-outside-toplevel  # Dynamic import for test

        client, workflow = workflow_client

        # Create chat session in database for the thread
        thread_id = "test-thread-invalid"
        await WorkflowChatSession.create(
            workflow=workflow,
            user=workflow.user,
            thread_id=thread_id,
            title="Test Session"
        )

        # Mock checkpointer to return stored interrupt (used for validation)
        mock_checkpointer = AsyncMock()
        mock_checkpointer.aget_tuple.return_value = MagicMock(
            checkpoint={
                "channel_values": {
                    "__interrupt__": [
                        {
                            "type": "clarification_questions",
                            "questions": [
                                {
                                    "question_id": "q_test123",
                                    "options": [
                                        {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                                        {"value": "outlook", "label": "Outlook", "is_wildcard": False},
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        )

        # Validation happens before .kiq() is called, so we just need to mock checkpointer
        with patch('seer.api.agents.workflow.router.get_checkpointer', return_value=mock_checkpointer):

            # Resume with invalid selection (validation error returned immediately)
            response = await client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": thread_id,
                    "answers": {
                        "answers": [
                            {
                                "question_id": "q_test123",
                                "selected_values": ["invalid_provider"],
                                "custom_input": None
                            }
                        ]
                    }
                }
            )

        assert response.status_code == 400
        data = response.json()
        # Check title or nested detail for error message
        assert data.get("title") == "Invalid selections" or "Invalid selections" in str(data.get("detail", {}))

    async def test_wildcard_requires_custom_input(
        self,
        workflow_client
    ):
        """Test that wildcard selection requires custom input (validation happens before async task)."""
        from seer.database.workflow_models import WorkflowChatSession  # pylint: disable=import-outside-toplevel  # Dynamic import for test

        client, workflow = workflow_client

        # Create chat session in database for the thread
        thread_id = "test-thread-wildcard"
        await WorkflowChatSession.create(
            workflow=workflow,
            user=workflow.user,
            thread_id=thread_id,
            title="Test Session"
        )

        # Mock checkpointer to return stored interrupt with wildcard (used for validation)
        mock_checkpointer = AsyncMock()
        mock_checkpointer.aget_tuple.return_value = MagicMock(
            checkpoint={
                "channel_values": {
                    "__interrupt__": [
                        {
                            "type": "clarification_questions",
                            "questions": [
                                {
                                    "question_id": "q_test123",
                                    "options": [
                                        {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                                        {"value": "other", "label": "Other", "is_wildcard": True},
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        )

        # Validation happens before .kiq() is called
        with patch('seer.api.agents.workflow.router.get_checkpointer', return_value=mock_checkpointer):

            # Resume with wildcard but no custom input (validation error returned immediately)
            response = await client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": thread_id,
                    "answers": {
                        "answers": [
                            {
                                "question_id": "q_test123",
                                "selected_values": ["other"],
                                "custom_input": None
                            }
                        ]
                    }
                }
            )

        assert response.status_code == 400
        data = response.json()
        # Check title or nested detail for error message
        assert data.get("title") == "Custom input required" or "custom input" in str(data.get("detail", {})).lower()

    async def test_wildcard_with_custom_input_succeeds(
        self,
        workflow_client,
        mock_task_result
    ):
        """Test that wildcard with custom input succeeds via async task."""
        from seer.database.workflow_models import WorkflowChatSession, ChatExecutionStatus  # pylint: disable=import-outside-toplevel  # Dynamic import for test

        client, workflow = workflow_client

        # Create chat session in database for the thread
        thread_id = "test-thread-wildcard-success"
        session = await WorkflowChatSession.create(
            workflow=workflow,
            user=workflow.user,
            thread_id=thread_id,
            title="Test Session"
        )

        # Mock checkpointer (used for validation)
        mock_checkpointer = AsyncMock()
        mock_checkpointer.aget_tuple.return_value = MagicMock(
            checkpoint={
                "channel_values": {
                    "__interrupt__": [
                        {
                            "type": "clarification_questions",
                            "questions": [
                                {
                                    "question_id": "q_test123",
                                    "options": [
                                        {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                                        {"value": "other", "label": "Other", "is_wildcard": True},
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        )

        # Mock agent to return completed response (used by the task)
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="I'll set up ProtonMail integration.")
            ]
        }

        # Capture task arguments
        captured_task_args = {}

        async def capture_kiq(**kwargs):
            captured_task_args.update(kwargs)
            return mock_task_result

        with patch('seer.api.agents.workflow.router.get_checkpointer', return_value=mock_checkpointer), \
             patch('seer.worker.tasks.chat.chat_resume_task') as mock_resume_task, \
             patch('seer.api.agents.workflow.router.get_stream_watermark', new_callable=AsyncMock, return_value="0"), \
             patch('seer.agents.nexus.stream_publisher.StreamPublisher') as mock_publisher_cls, \
             patch('seer.api.agents.workflow.router.stream_events_sse', new=_empty_sse):

            mock_publisher_cls.return_value = AsyncMock()
            mock_resume_task.kiq = AsyncMock(side_effect=capture_kiq)

            # Resume with wildcard and custom input (batch format)
            response = await client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": thread_id,
                    "answers": {
                        "answers": [
                            {
                                "question_id": "q_test123",
                                "selected_values": ["other"],
                                "custom_input": "ProtonMail"
                            }
                        ]
                    }
                }
            )

            assert response.status_code == 200

        # Simulate background task execution
        with patch('seer.worker.tasks.chat.create_nexus_chat_agent', return_value=mock_agent), \
             patch('seer.worker.tasks.chat.get_checkpointer', return_value=mock_checkpointer):

            from seer.worker.tasks.chat import chat_resume_task
            await chat_resume_task(
                session_id=session.id,
                user_id=workflow.user_id,
                thread_id=thread_id,
                resume_command_data=captured_task_args["resume_command_data"],
                workflow_id=workflow.id,
            )

        # Verify session completed
        await session.refresh_from_db()
        assert session.current_execution_status == ChatExecutionStatus.COMPLETED

    async def test_multi_choice_question(
        self,
        workflow_client,
        mock_task_result
    ):
        """Test multi-choice clarification question via async task."""
        from seer.database.workflow_models import WorkflowChatSession, ChatExecutionStatus  # pylint: disable=import-outside-toplevel

        client, workflow = workflow_client

        # Use batch format with multi-choice question
        interrupt_data = {
            "type": "clarification_questions",
            "questions": [
                {
                    "question_id": "q_multi123",
                    "question": "Which integrations to enable?",
                    "question_type": "multi_choice",
                    "options": [
                        {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                        {"value": "slack", "label": "Slack", "is_wildcard": False},
                        {"value": "github", "label": "GitHub", "is_wildcard": False},
                    ],
                    "min_selections": 1,
                    "max_selections": 3,
                    "reasoning": "User wants multiple integrations"
                }
            ]
        }

        # Mock agent to return multi-choice interrupt (used by the task)
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="Which integrations?")
            ],
            "__interrupt__": [interrupt_data]
        }

        # Capture task arguments
        captured_task_args = {}

        async def capture_kiq(**kwargs):
            captured_task_args.update(kwargs)
            return mock_task_result

        with patch('seer.api.agents.workflow.router.get_checkpointer') as mock_get_checkpointer, \
             patch('seer.worker.tasks.chat.chat_execution_task') as mock_chat_task, \
             patch('seer.api.agents.workflow.router.get_stream_watermark', new_callable=AsyncMock, return_value="0"), \
             patch('seer.agents.nexus.stream_publisher.StreamPublisher') as mock_publisher_cls, \
             patch('seer.api.agents.workflow.router.stream_events_sse', new=_empty_sse):

            mock_publisher_cls.return_value = AsyncMock()
            mock_get_checkpointer.return_value = AsyncMock()
            mock_chat_task.kiq = AsyncMock(side_effect=capture_kiq)

            # Send chat message (async_mode=True by default)
            response = await client.post(
                f"/nexus/{workflow.workflow_id}/chat",
                json={
                    "message": "Enable integrations"
                }
            )

            assert response.status_code == 200
            session_id = captured_task_args["session_id"]

        # Simulate background task execution
        with patch('seer.worker.tasks.chat.create_nexus_chat_agent', return_value=mock_agent), \
             patch('seer.worker.tasks.chat.get_checkpointer') as mock_task_checkpointer:

            mock_task_checkpointer.return_value = AsyncMock()

            from seer.worker.tasks.chat import chat_execution_task
            await chat_execution_task(
                session_id=session_id,
                user_id=workflow.user_id,
                message="Enable integrations",
                workflow_id=workflow.id,
            )

        # Verify session has multi-choice interrupt
        session = await WorkflowChatSession.get(id=session_id)
        assert session.current_execution_status == ChatExecutionStatus.INTERRUPTED
        assert session.pending_interrupt_type == "clarification_questions"

        # Verify multi-choice question in interrupt data
        stored_interrupt = session.pending_interrupt_data
        assert stored_interrupt is not None
        questions = stored_interrupt.get("questions", [])
        assert len(questions) == 1
        assert questions[0].get("question_type") == "multi_choice"

    async def test_multiple_questions_at_once(
        self,
        workflow_client,
        mock_task_result
    ):
        """Test asking multiple questions at once (batch mode)."""
        from seer.database.workflow_models import WorkflowChatSession, ChatExecutionStatus  # pylint: disable=import-outside-toplevel

        client, workflow = workflow_client

        # Batch format with multiple questions
        interrupt_data = {
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
                    "reasoning": "Need email config"
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
                    "reasoning": "Need notification config"
                }
            ]
        }

        # Mock agent to return batch interrupt
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="I have a couple of questions.")
            ],
            "__interrupt__": [interrupt_data]
        }

        captured_task_args = {}

        async def capture_kiq(**kwargs):
            captured_task_args.update(kwargs)
            return mock_task_result

        with patch('seer.api.agents.workflow.router.get_checkpointer') as mock_get_checkpointer, \
             patch('seer.worker.tasks.chat.chat_execution_task') as mock_chat_task, \
             patch('seer.api.agents.workflow.router.get_stream_watermark', new_callable=AsyncMock, return_value="0"), \
             patch('seer.agents.nexus.stream_publisher.StreamPublisher') as mock_publisher_cls, \
             patch('seer.api.agents.workflow.router.stream_events_sse', new=_empty_sse):

            mock_publisher_cls.return_value = AsyncMock()
            mock_get_checkpointer.return_value = AsyncMock()
            mock_chat_task.kiq = AsyncMock(side_effect=capture_kiq)

            response = await client.post(
                f"/nexus/{workflow.workflow_id}/chat",
                json={
                    "message": "Set up email and notifications"
                }
            )

            assert response.status_code == 200
            session_id = captured_task_args["session_id"]

        # Simulate task execution
        with patch('seer.worker.tasks.chat.create_nexus_chat_agent', return_value=mock_agent), \
             patch('seer.worker.tasks.chat.get_checkpointer') as mock_task_checkpointer:

            mock_task_checkpointer.return_value = AsyncMock()

            from seer.worker.tasks.chat import chat_execution_task
            await chat_execution_task(
                session_id=session_id,
                user_id=workflow.user_id,
                message="Set up email and notifications",
                workflow_id=workflow.id,
            )

        # Verify session has batch questions
        session = await WorkflowChatSession.get(id=session_id)
        assert session.current_execution_status == ChatExecutionStatus.INTERRUPTED
        assert session.pending_interrupt_type == "clarification_questions"

        questions = session.pending_interrupt_data.get("questions", [])
        assert len(questions) == 2
        assert questions[0]["question_id"] == "q_email"
        assert questions[1]["question_id"] == "q_notify"

        # Now test resuming with batch answers
        mock_checkpointer = AsyncMock()
        mock_checkpointer.aget_tuple.return_value = MagicMock(
            checkpoint={
                "channel_values": {
                    "__interrupt__": [interrupt_data]
                }
            }
        )

        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="Great! Setting up Gmail and Slack notifications.")
            ]
        }

        async def capture_resume_kiq(**kwargs):
            captured_task_args.clear()
            captured_task_args.update(kwargs)
            return mock_task_result

        with patch('seer.api.agents.workflow.router.get_checkpointer', return_value=mock_checkpointer), \
             patch('seer.worker.tasks.chat.chat_resume_task') as mock_resume_task, \
             patch('seer.api.agents.workflow.router.get_stream_watermark', new_callable=AsyncMock, return_value="0"), \
             patch('seer.agents.nexus.stream_publisher.StreamPublisher') as mock_publisher_cls_resume, \
             patch('seer.api.agents.workflow.router.stream_events_sse', new=_empty_sse):

            mock_publisher_cls_resume.return_value = AsyncMock()
            mock_resume_task.kiq = AsyncMock(side_effect=capture_resume_kiq)

            response = await client.post(
                f"/nexus/{workflow.workflow_id}/chat/resume",
                json={
                    "thread_id": session.thread_id,
                    "answers": {
                        "answers": [
                            {
                                "question_id": "q_email",
                                "selected_values": ["gmail"],
                                "custom_input": None
                            },
                            {
                                "question_id": "q_notify",
                                "selected_values": ["slack"],
                                "custom_input": None
                            }
                        ]
                    }
                }
            )

            assert response.status_code == 200

        # Simulate resume task
        with patch('seer.worker.tasks.chat.create_nexus_chat_agent', return_value=mock_agent), \
             patch('seer.worker.tasks.chat.get_checkpointer', return_value=mock_checkpointer):

            from seer.worker.tasks.chat import chat_resume_task
            await chat_resume_task(
                session_id=session.id,
                user_id=workflow.user_id,
                thread_id=session.thread_id,
                resume_command_data=captured_task_args["resume_command_data"],
                workflow_id=workflow.id,
            )

        await session.refresh_from_db()
        assert session.current_execution_status == ChatExecutionStatus.COMPLETED
