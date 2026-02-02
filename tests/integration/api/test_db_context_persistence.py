"""
Integration tests for database-backed context persistence.

Tests that workflow state, proposals, and user lookups are persisted in the database
and survive across restarts.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.mark.integration
@pytest.mark.asyncio
class TestDBContextPersistence:
    """Integration tests for database-backed context."""

    @pytest.fixture
    async def workflow_client(self, mock_app: FastAPI, db_engine, test_user, test_workflow):  # pylint: disable=unused-argument
        """Create authenticated API client with workflow router."""
        from seer.api.agents.workflow import router  # pylint: disable=import-outside-toplevel

        # Add router to test app
        mock_app.include_router(router)

        # Mock authentication to inject test_user
        from fastapi import Request  # pylint: disable=import-outside-toplevel
        async def mock_auth_middleware(request: Request, call_next):
            request.state.db_user = test_user
            response = await call_next(request)
            return response

        mock_app.middleware("http")(mock_auth_middleware)

        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, test_workflow

    async def test_workflow_state_persistence(self, workflow_client):
        """Test that workflow state is saved to database and can be retrieved."""
        from seer.database.workflow_models import WorkflowChatSession
        from unittest.mock import patch

        client, workflow = workflow_client

        workflow_state = {
            "nodes": [{"id": "node1", "type": "tool", "data": {"label": "Test Node"}}],
            "edges": [{"id": "edge1", "source": "node1", "target": "node2"}]
        }

        # Mock the taskiq task's kiq method to avoid broker connection
        mock_task_result = MagicMock()
        mock_task_result.task_id = "mock-task-id-123"

        with patch('seer.api.agents.workflow.router.get_checkpointer') as mock_checkpointer, \
             patch('seer.worker.tasks.chat.chat_execution_task') as mock_chat_task:

            mock_checkpointer.return_value = AsyncMock()
            mock_chat_task.kiq = AsyncMock(return_value=mock_task_result)

            # Send chat message with workflow state (async mode - the default)
            response = await client.post(
                f"/nexus/{workflow.workflow_id}/chat",
                json={
                    "message": "Test message",
                    "workflow_state": workflow_state
                }
            )

            assert response.status_code == 200
            data = response.json()
            thread_id = data["thread_id"]

            # Verify workflow state was saved to database (happens before async task is enqueued)
            session = await WorkflowChatSession.get_or_none(thread_id=thread_id)
            assert session is not None
            assert session.current_workflow_state is not None
            assert session.current_workflow_state["nodes"] == workflow_state["nodes"]
            assert session.current_workflow_state["edges"] == workflow_state["edges"]

    async def test_proposal_creation_by_thread(self, workflow_client):
        """Test that proposals are created with thread_id and can be queried."""
        from seer.database.workflow_models import WorkflowProposal
        from seer.api.agents.workflow.services import create_chat_session
        from seer.database import User

        client, workflow = workflow_client

        user = await User.get(id=workflow.user_id)
        session = await create_chat_session(
            workflow=workflow,
            user=user,
            thread_id="test-thread-123",
            title="Test Session"
        )

        # Create a proposal with thread_id (simulating what submit_workflow_spec does)
        spec = {
            "version": "v2",
            "nodes": [
                {
                    "id": "test_node",
                    "type": "tool",
                    "tool": "discord_send_message",
                    "inputs": {
                        "channel_id": "123",
                        "message": "test"
                    }
                }
            ],
            "edges": []
        }

        proposal = await WorkflowProposal.create(
            workflow=workflow,
            session=session,
            created_by=user,
            summary="Test workflow",
            spec=spec,
            status=WorkflowProposal.STATUS_PENDING,
            thread_id=session.thread_id,
        )

        # Verify proposal can be queried by thread_id
        proposals = await WorkflowProposal.filter(
            thread_id=session.thread_id,
            status=WorkflowProposal.STATUS_PENDING
        ).all()

        assert len(proposals) == 1
        assert proposals[0].thread_id == session.thread_id
        assert proposals[0].workflow_id == workflow.id
        assert proposals[0].session_id == session.id
        assert proposals[0].id == proposal.id

    async def test_user_lookup_via_session(self, workflow_client):
        """Test that user can be looked up via session relationship."""
        from seer.agents.nexus.context import get_user_for_thread
        from seer.api.agents.workflow.services import create_chat_session
        from seer.database import User

        client, workflow = workflow_client

        user = await User.get(id=workflow.user_id)
        session = await create_chat_session(
            workflow=workflow,
            user=user,
            thread_id="test-thread-user-lookup",
            title="User Lookup Test"
        )

        # Look up user by thread_id
        retrieved_user = await get_user_for_thread(session.thread_id)

        assert retrieved_user is not None
        assert retrieved_user.id == user.id

    async def test_thread_isolation(self, workflow_client):
        """Test that different threads maintain isolated state."""
        from seer.database.workflow_models import WorkflowChatSession
        from seer.api.agents.workflow.services import create_chat_session
        from seer.database import User

        client, workflow = workflow_client

        user = await User.get(id=workflow.user_id)

        # Create two sessions with different workflow states
        session1 = await create_chat_session(
            workflow=workflow,
            user=user,
            thread_id="test-thread-isolation-1",
            title="Session 1"
        )
        session1.current_workflow_state = {
            "nodes": [{"id": "node1"}],
            "edges": []
        }
        await session1.save(update_fields=['current_workflow_state'])

        session2 = await create_chat_session(
            workflow=workflow,
            user=user,
            thread_id="test-thread-isolation-2",
            title="Session 2"
        )
        session2.current_workflow_state = {
            "nodes": [{"id": "node2"}],
            "edges": []
        }
        await session2.save(update_fields=['current_workflow_state'])

        # Verify isolation
        retrieved_session1 = await WorkflowChatSession.get(thread_id=session1.thread_id)
        retrieved_session2 = await WorkflowChatSession.get(thread_id=session2.thread_id)

        assert retrieved_session1.current_workflow_state["nodes"][0]["id"] == "node1"
        assert retrieved_session2.current_workflow_state["nodes"][0]["id"] == "node2"

    async def test_workflow_state_retrieval_by_tools(self, workflow_client):
        """Test that tools can retrieve workflow state from database."""
        from seer.agents.nexus.context import get_workflow_state_for_thread
        from seer.api.agents.workflow.services import create_chat_session
        from seer.database import User

        client, workflow = workflow_client

        user = await User.get(id=workflow.user_id)
        session = await create_chat_session(
            workflow=workflow,
            user=user,
            thread_id="test-thread-state-retrieval",
            title="State Retrieval Test"
        )

        # Set workflow state
        workflow_state = {
            "nodes": [{"id": "test_node", "type": "tool"}],
            "edges": [{"id": "edge1", "source": "test_node", "target": "end"}]
        }
        session.current_workflow_state = workflow_state
        await session.save(update_fields=['current_workflow_state'])

        # Retrieve via context function
        retrieved_state = await get_workflow_state_for_thread(session.thread_id)

        assert retrieved_state is not None
        assert retrieved_state["nodes"] == workflow_state["nodes"]
        assert retrieved_state["edges"] == workflow_state["edges"]
