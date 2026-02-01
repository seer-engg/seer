"""
Integration tests for clarification question flow.

Tests the full flow: chat -> interrupt -> resume with clarification questions.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


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

    @patch('seer.api.agents.workflow.router.create_nexus_chat_agent')
    @patch('seer.api.agents.workflow.router.get_checkpointer')
    async def test_interrupt_triggered_by_clarification_tool(
        self,
        mock_get_checkpointer,
        mock_create_agent,
        workflow_client
    ):
        """Test that clarification question triggers interrupt."""
        client, workflow = workflow_client

        # Mock agent to return interrupt
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="I need to know which email provider to use.")
            ],
            "__interrupt__": [
                {
                    "type": "clarification_question",
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
        mock_create_agent.return_value = mock_agent

        # Mock checkpointer
        mock_checkpointer = AsyncMock()
        mock_get_checkpointer.return_value = mock_checkpointer

        # Send chat message
        response = await client.post(
            f"/nexus/{workflow.workflow_id}/chat",
            json={
                "message": "Set up email integration",
                "workflow_state": {"nodes": [], "edges": []}
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify interrupt is set
        assert data["interrupt_required"] is True
        assert data["interrupt_data"] is not None
        assert data["interrupt_data"]["type"] == "clarification_question"

        # Verify structured question in interrupt_data
        question = data["interrupt_data"]["clarification_question"]
        assert question["question_id"] == "q_test123"
        assert question["question"] == "Which email provider should we use?"
        assert question["question_type"] == "single_choice"
        assert len(question["options"]) == 3
        assert question["min_selections"] == 1

        # Verify wildcard option
        wildcard_opts = [opt for opt in question["options"] if opt["is_wildcard"]]
        assert len(wildcard_opts) == 1
        assert wildcard_opts[0]["value"] == "other"

    @patch('seer.api.agents.workflow.router.create_nexus_chat_agent')
    @patch('seer.api.agents.workflow.router.get_checkpointer')
    async def test_resume_with_valid_answer(
        self,
        mock_get_checkpointer,
        mock_create_agent,
        workflow_client
    ):
        """Test resuming with valid clarification answer."""
        client, workflow = workflow_client

        # Mock checkpointer to return stored interrupt
        mock_checkpointer = AsyncMock()
        mock_state = MagicMock()
        mock_state.values = {
            "__interrupt__": [
                {
                    "type": "clarification_question",
                    "question_id": "q_test123",
                    "options": [
                        {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                        {"value": "outlook", "label": "Outlook", "is_wildcard": False},
                    ]
                }
            ]
        }
        mock_checkpointer.aget.return_value = mock_state
        mock_get_checkpointer.return_value = mock_checkpointer

        # Mock agent to return continued response
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="Great! I'll set up Gmail integration.")
            ]
        }
        mock_create_agent.return_value = mock_agent

        # Resume with answer
        response = await client.post(
            f"/nexus/{workflow.workflow_id}/chat/resume",
            json={
                "thread_id": "test-thread-123",
                "answer": {
                    "question_id": "q_test123",
                    "selected_values": ["gmail"],
                    "custom_input": None
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["interrupt_required"] is False

    @patch('seer.api.agents.workflow.router.create_nexus_chat_agent')
    @patch('seer.api.agents.workflow.router.get_checkpointer')
    async def test_resume_with_invalid_selection(
        self,
        mock_get_checkpointer,
        mock_create_agent,
        workflow_client
    ):
        """Test that invalid selections are rejected."""
        client, workflow = workflow_client

        # Mock checkpointer to return stored interrupt
        mock_checkpointer = AsyncMock()
        mock_state = MagicMock()
        mock_state.values = {
            "__interrupt__": [
                {
                    "type": "clarification_question",
                    "question_id": "q_test123",
                    "options": [
                        {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                        {"value": "outlook", "label": "Outlook", "is_wildcard": False},
                    ]
                }
            ]
        }
        mock_checkpointer.aget.return_value = mock_state
        mock_get_checkpointer.return_value = mock_checkpointer

        # Mock agent
        mock_agent = AsyncMock()
        mock_create_agent.return_value = mock_agent

        # Resume with invalid selection
        response = await client.post(
            f"/nexus/{workflow.workflow_id}/chat/resume",
            json={
                "thread_id": "test-thread-123",
                "answer": {
                    "question_id": "q_test123",
                    "selected_values": ["invalid_provider"],
                    "custom_input": None
                }
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "Invalid selections" in data["detail"]

    @patch('seer.api.agents.workflow.router.create_nexus_chat_agent')
    @patch('seer.api.agents.workflow.router.get_checkpointer')
    async def test_wildcard_requires_custom_input(
        self,
        mock_get_checkpointer,
        mock_create_agent,
        workflow_client
    ):
        """Test that wildcard selection requires custom input."""
        client, workflow = workflow_client

        # Mock checkpointer to return stored interrupt with wildcard
        mock_checkpointer = AsyncMock()
        mock_state = MagicMock()
        mock_state.values = {
            "__interrupt__": [
                {
                    "type": "clarification_question",
                    "question_id": "q_test123",
                    "options": [
                        {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                        {"value": "other", "label": "Other", "is_wildcard": True},
                    ]
                }
            ]
        }
        mock_checkpointer.aget.return_value = mock_state
        mock_get_checkpointer.return_value = mock_checkpointer

        # Mock agent
        mock_agent = AsyncMock()
        mock_create_agent.return_value = mock_agent

        # Resume with wildcard but no custom input
        response = await client.post(
            f"/nexus/{workflow.workflow_id}/chat/resume",
            json={
                "thread_id": "test-thread-123",
                "answer": {
                    "question_id": "q_test123",
                    "selected_values": ["other"],
                    "custom_input": None
                }
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "Custom input required" in data["detail"] or "custom input" in data["detail"].lower()

    @patch('seer.api.agents.workflow.router.create_nexus_chat_agent')
    @patch('seer.api.agents.workflow.router.get_checkpointer')
    async def test_wildcard_with_custom_input_succeeds(
        self,
        mock_get_checkpointer,
        mock_create_agent,
        workflow_client
    ):
        """Test that wildcard with custom input succeeds."""
        client, workflow = workflow_client

        # Mock checkpointer
        mock_checkpointer = AsyncMock()
        mock_state = MagicMock()
        mock_state.values = {
            "__interrupt__": [
                {
                    "type": "clarification_question",
                    "question_id": "q_test123",
                    "options": [
                        {"value": "gmail", "label": "Gmail", "is_wildcard": False},
                        {"value": "other", "label": "Other", "is_wildcard": True},
                    ]
                }
            ]
        }
        mock_checkpointer.aget.return_value = mock_state
        mock_get_checkpointer.return_value = mock_checkpointer

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="I'll set up ProtonMail integration.")
            ]
        }
        mock_create_agent.return_value = mock_agent

        # Resume with wildcard and custom input
        response = await client.post(
            f"/nexus/{workflow.workflow_id}/chat/resume",
            json={
                "thread_id": "test-thread-123",
                "answer": {
                    "question_id": "q_test123",
                    "selected_values": ["other"],
                    "custom_input": "ProtonMail"
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    @patch('seer.api.agents.workflow.router.create_nexus_chat_agent')
    @patch('seer.api.agents.workflow.router.get_checkpointer')
    async def test_multi_choice_question(
        self,
        mock_get_checkpointer,
        mock_create_agent,
        workflow_client
    ):
        """Test multi-choice clarification question."""
        client, workflow = workflow_client

        # Mock agent to return multi-choice interrupt
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="Which integrations?")
            ],
            "__interrupt__": [
                {
                    "type": "clarification_question",
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
        mock_create_agent.return_value = mock_agent
        mock_checkpointer = AsyncMock()
        mock_get_checkpointer.return_value = mock_checkpointer

        # Send chat message
        response = await client.post(
            f"/nexus/{workflow.workflow_id}/chat",
            json={
                "message": "Enable integrations",
                "workflow_state": {"nodes": [], "edges": []}
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify multi-choice question
        question = data["interrupt_data"]["clarification_question"]
        assert question["question_type"] == "multi_choice"
        assert question["min_selections"] == 1
        assert question["max_selections"] == 3
