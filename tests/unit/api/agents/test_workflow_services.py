"""
Unit tests for workflow agent services.

Tests:
- workflow_state_from_spec: Graph snapshot generation
- _workflow_state_from_spec: Internal conversion
- workflow_state_snapshot: Snapshot from workflow
- _get_workflow / get_workflow: Workflow retrieval and authorization
- Chat session services: create, get, list, save messages, load history
- Proposal services: create, get, accept, reject
- Discovery session services: create, get, link
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# =============================================================================
# Fixtures
# =============================================================================

# Note: mock_user and mock_workflow fixtures are provided by tests/unit/conftest.py


@pytest.fixture
def mock_chat_session():
    """Create a mock chat session for testing."""
    from seer.database import WorkflowChatSession
    session = MagicMock(spec=WorkflowChatSession)
    session.id = 1
    session.thread_id = "thread_123"
    session.title = "Test Session"
    session.workflow = MagicMock()
    session.user = MagicMock()
    session.created_at = datetime.now(timezone.utc)
    session.updated_at = datetime.now(timezone.utc)
    session.fetch_related = AsyncMock()
    session.save = AsyncMock()
    return session


@pytest.fixture
def mock_chat_message():
    """Create a mock chat message for testing."""
    from seer.database import WorkflowChatMessage
    message = MagicMock(spec=WorkflowChatMessage)
    message.id = 1
    message.session_id = 1
    message.role = "user"
    message.content = "Hello"
    message.thinking = None
    message.suggested_edits = None
    message.metadata = None
    message.created_at = datetime.now(timezone.utc)
    return message


@pytest.fixture
def mock_proposal():
    """Create a mock workflow proposal for testing."""
    from seer.database import WorkflowProposal
    proposal = MagicMock(spec=WorkflowProposal)
    proposal.id = 1
    proposal.workflow_id = 1
    proposal.status = WorkflowProposal.STATUS_PENDING
    proposal.summary = "Test proposal"
    proposal.spec = {"version": "2", "nodes": [], "edges": []}
    proposal.workflow = AsyncMock(return_value=MagicMock())
    proposal.save = AsyncMock()
    return proposal


@pytest.fixture
def mock_draft_version():
    """Create a mock draft version for testing."""
    from seer.database import WorkflowVersion, WorkflowVersionStatus
    version = MagicMock(spec=WorkflowVersion)
    version.id = 1
    version.status = WorkflowVersionStatus.DRAFT
    version.spec = {"version": "2", "nodes": [], "edges": []}
    version.save = AsyncMock()
    return version


@pytest.fixture
def sample_spec_dict():
    """Sample workflow spec dict."""
    return {
        "version": "2",
        "nodes": [
            {
                "id": "n1",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {},
                "meta": {"label": "Node 1", "position": {"x": 100, "y": 100}},
            },
            {
                "id": "n2",
                "type": "llm",
                "model": "gpt-4",
                "prompt": "Test",
                "output": {},
                "meta": {"label": "Node 2", "position": {"x": 200, "y": 200}},
            },
        ],
        "edges": [],
    }


@pytest.fixture
def minimal_spec_dict():
    """Minimal workflow spec dict."""
    return {
        "version": "2",
        "nodes": [],
        "edges": [],
    }


# =============================================================================
# Workflow State From Spec Tests
# =============================================================================


@pytest.mark.unit
class TestWorkflowStateFromSpec:
    """Tests for workflow_state_from_spec function."""

    def test_workflow_state_from_spec_basic(self, sample_spec_dict):
        """Test converting spec to workflow state."""
        from seer.api.agents.workflow.services import workflow_state_from_spec

        result = workflow_state_from_spec(sample_spec_dict)

        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 2

    def test_workflow_state_from_spec_empty(self, minimal_spec_dict):
        """Test converting empty spec."""
        from seer.api.agents.workflow.services import workflow_state_from_spec

        result = workflow_state_from_spec(minimal_spec_dict)

        assert result["nodes"] == []
        assert result["edges"] == []

    def test_workflow_state_from_spec_invalid_input(self):
        """Test handling invalid input."""
        from seer.api.agents.workflow.services import workflow_state_from_spec

        result = workflow_state_from_spec("not a dict")

        assert result == {"nodes": [], "edges": []}

    def test_workflow_state_from_spec_none_input(self):
        """Test handling None input."""
        from seer.api.agents.workflow.services import workflow_state_from_spec

        result = workflow_state_from_spec(None)

        assert result == {"nodes": [], "edges": []}


# =============================================================================
# Internal Workflow State Tests
# =============================================================================


@pytest.mark.unit
class TestInternalWorkflowState:
    """Tests for _workflow_state_from_spec internal function."""

    def test_node_extraction(self, sample_spec_dict):
        """Test that nodes are correctly extracted."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        result = _workflow_state_from_spec(sample_spec_dict)

        node1 = result["nodes"][0]
        assert node1["id"] == "n1"
        assert node1["type"] == "tool"
        assert node1["data"]["label"] == "Node 1"

    def test_position_extraction(self, sample_spec_dict):
        """Test that positions are extracted from meta."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        result = _workflow_state_from_spec(sample_spec_dict)

        node1 = result["nodes"][0]
        assert "position" in node1
        assert node1["position"]["x"] == 100
        assert node1["position"]["y"] == 100

    def test_edge_generation(self, sample_spec_dict):
        """Test that edges are generated between sequential nodes."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        result = _workflow_state_from_spec(sample_spec_dict)

        assert len(result["edges"]) == 1
        edge = result["edges"][0]
        assert edge["source"] == "n1"
        assert edge["target"] == "n2"

    def test_node_without_meta(self):
        """Test handling nodes without meta field."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {
            "nodes": [
                {"id": "n1", "type": "tool"},
            ]
        }

        result = _workflow_state_from_spec(spec)

        node = result["nodes"][0]
        assert node["data"]["label"] == "n1"
        assert "position" not in node

    def test_node_with_invalid_meta(self):
        """Test handling nodes with invalid meta type."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {
            "nodes": [
                {"id": "n1", "type": "tool", "meta": "not a dict"},
            ]
        }

        result = _workflow_state_from_spec(spec)

        node = result["nodes"][0]
        assert node["data"]["label"] == "n1"

    def test_node_with_empty_meta(self):
        """Test handling nodes with empty meta dict."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {
            "nodes": [
                {"id": "n1", "type": "tool", "meta": {}},
            ]
        }

        result = _workflow_state_from_spec(spec)

        node = result["nodes"][0]
        assert node["data"]["label"] == "n1"
        assert "position" not in node

    def test_nodes_not_a_list(self):
        """Test handling when nodes field is not a list."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {"nodes": "not a list"}

        result = _workflow_state_from_spec(spec)

        assert result["nodes"] == []
        assert result["edges"] == []

    def test_node_not_a_dict(self):
        """Test handling when a node is not a dict."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {"nodes": ["not a dict", {"id": "n1", "type": "tool"}]}

        result = _workflow_state_from_spec(spec)

        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "n1"

    def test_position_with_missing_coordinates(self):
        """Test position defaults when x/y are missing."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {
            "nodes": [
                {"id": "n1", "type": "tool", "meta": {"position": {}}},
            ]
        }

        result = _workflow_state_from_spec(spec)

        node = result["nodes"][0]
        assert node["position"]["x"] == 0
        assert node["position"]["y"] == 0

    def test_multiple_edges_for_multiple_nodes(self):
        """Test edge generation for multiple sequential nodes."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {
            "nodes": [
                {"id": "n1", "type": "tool"},
                {"id": "n2", "type": "tool"},
                {"id": "n3", "type": "tool"},
            ]
        }

        result = _workflow_state_from_spec(spec)

        assert len(result["edges"]) == 2
        assert result["edges"][0]["source"] == "n1"
        assert result["edges"][0]["target"] == "n2"
        assert result["edges"][1]["source"] == "n2"
        assert result["edges"][1]["target"] == "n3"


# =============================================================================
# Workflow State Snapshot Tests
# =============================================================================


@pytest.mark.unit
class TestWorkflowStateSnapshot:
    """Tests for workflow_state_snapshot function."""

    @pytest.mark.asyncio
    async def test_workflow_state_snapshot_with_draft(self, mock_workflow, mock_draft_version, sample_spec_dict):
        """Test getting snapshot from workflow with draft."""
        mock_draft_version.spec = sample_spec_dict

        with patch("seer.api.agents.workflow.services._get_draft_version", new_callable=AsyncMock) as mock_get_draft:
            mock_get_draft.return_value = mock_draft_version

            from seer.api.agents.workflow.services import workflow_state_snapshot

            result = await workflow_state_snapshot(mock_workflow)

            assert len(result["nodes"]) == 2
            mock_get_draft.assert_called_once_with(mock_workflow, create_if_missing=False)

    @pytest.mark.asyncio
    async def test_workflow_state_snapshot_no_draft(self, mock_workflow):
        """Test getting snapshot from workflow without draft."""
        with patch("seer.api.agents.workflow.services._get_draft_version", new_callable=AsyncMock) as mock_get_draft:
            mock_get_draft.return_value = None

            from seer.api.agents.workflow.services import workflow_state_snapshot

            result = await workflow_state_snapshot(mock_workflow)

            assert result == {"nodes": [], "edges": []}

    @pytest.mark.asyncio
    async def test_workflow_state_snapshot_draft_with_invalid_spec(self, mock_workflow, mock_draft_version):
        """Test snapshot when draft has non-dict spec."""
        mock_draft_version.spec = "not a dict"

        with patch("seer.api.agents.workflow.services._get_draft_version", new_callable=AsyncMock) as mock_get_draft:
            mock_get_draft.return_value = mock_draft_version

            from seer.api.agents.workflow.services import workflow_state_snapshot

            result = await workflow_state_snapshot(mock_workflow)

            assert result == {"nodes": [], "edges": []}


# =============================================================================
# Get Workflow Tests
# =============================================================================


@pytest.mark.unit
class TestGetWorkflow:
    """Tests for _get_workflow and get_workflow functions."""

    @pytest.mark.asyncio
    async def test_get_workflow_success(self, mock_user, mock_workflow):
        """Test successful workflow retrieval."""
        from seer.database import Workflow

        with patch.object(Workflow, "filter") as mock_filter:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=mock_workflow)
            mock_filter.return_value = mock_query

            from seer.api.agents.workflow.services import get_workflow

            result = await get_workflow(mock_user, "wf_1")

            assert result == mock_workflow
            mock_filter.assert_called_once_with(id=1, user=mock_user)

    @pytest.mark.asyncio
    async def test_get_workflow_not_found(self, mock_user):
        """Test workflow not found raises HTTPException."""
        from seer.database import Workflow

        with patch.object(Workflow, "filter") as mock_filter:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=None)
            mock_filter.return_value = mock_query

            from seer.api.agents.workflow.services import get_workflow

            with pytest.raises(HTTPException) as exc_info:
                await get_workflow(mock_user, "wf_999")

            assert exc_info.value.status_code == 404
            assert "not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_workflow_invalid_id_format(self, mock_user):
        """Test invalid workflow ID format raises HTTPException."""
        from seer.api.agents.workflow.services import get_workflow

        with pytest.raises(HTTPException) as exc_info:
            await get_workflow(mock_user, "invalid_id")

        assert exc_info.value.status_code == 400
        assert "Invalid workflow id format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_workflow_empty_id(self, mock_user):
        """Test empty workflow ID raises HTTPException."""
        from seer.api.agents.workflow.services import get_workflow

        with pytest.raises(HTTPException) as exc_info:
            await get_workflow(mock_user, "")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_get_workflow_non_numeric_suffix(self, mock_user):
        """Test non-numeric suffix raises HTTPException."""
        from seer.api.agents.workflow.services import get_workflow

        with pytest.raises(HTTPException) as exc_info:
            await get_workflow(mock_user, "wf_abc")

        assert exc_info.value.status_code == 400


# =============================================================================
# Chat Session Services Tests
# =============================================================================


@pytest.mark.unit
class TestChatSessionServices:
    """Tests for chat session management functions."""

    @pytest.mark.asyncio
    async def test_create_chat_session(self, mock_workflow, mock_user, mock_chat_session):
        """Test creating a new chat session."""
        from seer.database import WorkflowChatSession

        with patch.object(WorkflowChatSession, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_chat_session

            from seer.api.agents.workflow.services import create_chat_session

            result = await create_chat_session(
                workflow=mock_workflow,
                user=mock_user,
                thread_id="thread_123",
                title="Test Session",
            )

            assert result == mock_chat_session
            mock_create.assert_called_once_with(
                workflow=mock_workflow,
                user=mock_user,
                thread_id="thread_123",
                title="Test Session",
            )
            mock_chat_session.fetch_related.assert_called_once_with("user")

    @pytest.mark.asyncio
    async def test_get_chat_session_found(self, mock_workflow, mock_chat_session):
        """Test getting an existing chat session."""
        from seer.database import WorkflowChatSession

        with patch.object(WorkflowChatSession, "filter") as mock_filter:
            mock_query = MagicMock()
            mock_query.prefetch_related = MagicMock(return_value=mock_query)
            mock_query.first = AsyncMock(return_value=mock_chat_session)
            mock_filter.return_value = mock_query

            from seer.api.agents.workflow.services import get_chat_session

            result = await get_chat_session(session_id=1, workflow=mock_workflow)

            assert result == mock_chat_session
            mock_filter.assert_called_once_with(id=1, workflow=mock_workflow)

    @pytest.mark.asyncio
    async def test_get_chat_session_not_found(self, mock_workflow):
        """Test getting a non-existent chat session raises HTTPException."""
        from seer.database import WorkflowChatSession

        with patch.object(WorkflowChatSession, "filter") as mock_filter:
            mock_query = MagicMock()
            mock_query.prefetch_related = MagicMock(return_value=mock_query)
            mock_query.first = AsyncMock(return_value=None)
            mock_filter.return_value = mock_query

            from seer.api.agents.workflow.services import get_chat_session

            with pytest.raises(HTTPException) as exc_info:
                await get_chat_session(session_id=999, workflow=mock_workflow)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_chat_session_by_thread_id_found(self, mock_workflow, mock_chat_session):
        """Test getting a chat session by thread ID."""
        from seer.database import WorkflowChatSession

        with patch.object(WorkflowChatSession, "filter") as mock_filter:
            mock_query = MagicMock()
            mock_query.prefetch_related = MagicMock(return_value=mock_query)
            mock_query.first = AsyncMock(return_value=mock_chat_session)
            mock_filter.return_value = mock_query

            from seer.api.agents.workflow.services import get_chat_session_by_thread_id

            result = await get_chat_session_by_thread_id(thread_id="thread_123", workflow=mock_workflow)

            assert result == mock_chat_session

    @pytest.mark.asyncio
    async def test_get_chat_session_by_thread_id_not_found(self, mock_workflow):
        """Test getting a non-existent chat session by thread ID returns None."""
        from seer.database import WorkflowChatSession

        with patch.object(WorkflowChatSession, "filter") as mock_filter:
            mock_query = MagicMock()
            mock_query.prefetch_related = MagicMock(return_value=mock_query)
            mock_query.first = AsyncMock(return_value=None)
            mock_filter.return_value = mock_query

            from seer.api.agents.workflow.services import get_chat_session_by_thread_id

            result = await get_chat_session_by_thread_id(thread_id="nonexistent", workflow=mock_workflow)

            assert result is None

    @pytest.mark.asyncio
    async def test_list_chat_sessions(self, mock_workflow, mock_user, mock_chat_session):
        """Test listing chat sessions."""
        from seer.database import WorkflowChatSession

        with patch.object(WorkflowChatSession, "filter") as mock_filter:
            mock_query = MagicMock()
            mock_query.prefetch_related = MagicMock(return_value=mock_query)
            mock_query.order_by = MagicMock(return_value=mock_query)
            mock_query.offset = MagicMock(return_value=mock_query)
            mock_query.limit = MagicMock(return_value=mock_query)
            mock_query.all = AsyncMock(return_value=[mock_chat_session])
            mock_filter.return_value = mock_query

            from seer.api.agents.workflow.services import list_chat_sessions

            result = await list_chat_sessions(workflow=mock_workflow, user=mock_user, limit=50, offset=0)

            assert len(result) == 1
            assert result[0] == mock_chat_session
            mock_query.order_by.assert_called_once_with("-created_at")


# =============================================================================
# Chat Message Services Tests
# =============================================================================


@pytest.mark.unit
class TestChatMessageServices:
    """Tests for chat message functions."""

    @pytest.mark.asyncio
    async def test_save_chat_message_success(self, mock_chat_session, mock_chat_message):
        """Test saving a chat message."""
        from seer.database import WorkflowChatSession, WorkflowChatMessage

        with patch.object(WorkflowChatSession, "get_or_none", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_chat_session

            with patch.object(WorkflowChatMessage, "create", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_chat_message

                from seer.api.agents.workflow.services import save_chat_message

                result = await save_chat_message(
                    session_id=1,
                    role="user",
                    content="Hello",
                )

                assert result == mock_chat_message
                mock_create.assert_called_once()
                mock_chat_session.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_chat_message_session_not_found(self):
        """Test saving message to non-existent session raises HTTPException."""
        from seer.database import WorkflowChatSession

        with patch.object(WorkflowChatSession, "get_or_none", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            from seer.api.agents.workflow.services import save_chat_message

            with pytest.raises(HTTPException) as exc_info:
                await save_chat_message(session_id=999, role="user", content="Hello")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_save_chat_message_with_metadata(self, mock_chat_session, mock_chat_message):
        """Test saving a chat message with metadata."""
        from seer.database import WorkflowChatSession, WorkflowChatMessage

        with patch.object(WorkflowChatSession, "get_or_none", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_chat_session

            with patch.object(WorkflowChatMessage, "create", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_chat_message

                from seer.api.agents.workflow.services import save_chat_message

                await save_chat_message(
                    session_id=1,
                    role="assistant",
                    content="Response",
                    thinking="Thinking about it...",
                    suggested_edits={"nodes": []},
                    metadata={"model": "gpt-4"},
                )

                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["thinking"] == "Thinking about it..."
                assert call_kwargs["suggested_edits"] == {"nodes": []}
                assert call_kwargs["metadata"] == {"model": "gpt-4"}

    @pytest.mark.asyncio
    async def test_load_chat_history(self, mock_chat_message):
        """Test loading chat history for a session."""
        from seer.database import WorkflowChatMessage

        with patch.object(WorkflowChatMessage, "filter") as mock_filter:
            mock_query = MagicMock()
            mock_query.prefetch_related = MagicMock(return_value=mock_query)
            mock_query.order_by = MagicMock(return_value=mock_query)
            mock_query.all = AsyncMock(return_value=[mock_chat_message])
            mock_filter.return_value = mock_query

            from seer.api.agents.workflow.services import load_chat_history

            result = await load_chat_history(session_id=1)

            assert len(result) == 1
            assert result[0] == mock_chat_message
            mock_query.order_by.assert_called_once_with("created_at")

    @pytest.mark.asyncio
    async def test_save_chat_message_idempotent_returns_existing_when_proposal_linked(
        self, mock_chat_session, mock_chat_message, mock_proposal
    ):
        """Test that save_chat_message returns existing message if proposal already linked (task retry scenario)."""
        from seer.database import WorkflowChatSession, WorkflowChatMessage

        with patch.object(WorkflowChatSession, "get_or_none", new_callable=AsyncMock) as mock_get_session:
            mock_get_session.return_value = mock_chat_session

            with patch.object(WorkflowChatMessage, "get_or_none", new_callable=AsyncMock) as mock_get_msg:
                # Simulate existing message already linked to proposal
                mock_get_msg.return_value = mock_chat_message

                with patch.object(WorkflowChatMessage, "create", new_callable=AsyncMock) as mock_create:
                    from seer.api.agents.workflow.services import save_chat_message

                    result = await save_chat_message(
                        session_id=1,
                        role="assistant",
                        content="New content that should be ignored",
                        proposal=mock_proposal,
                    )

                    # Should return existing message, not create new one
                    assert result == mock_chat_message
                    mock_get_msg.assert_called_once_with(proposal=mock_proposal)
                    mock_create.assert_not_called()
                    # Session should not be updated on idempotent return
                    mock_chat_session.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_chat_message_creates_when_proposal_not_linked(
        self, mock_chat_session, mock_chat_message, mock_proposal
    ):
        """Test that save_chat_message creates new message if proposal not yet linked."""
        from seer.database import WorkflowChatSession, WorkflowChatMessage

        with patch.object(WorkflowChatSession, "get_or_none", new_callable=AsyncMock) as mock_get_session:
            mock_get_session.return_value = mock_chat_session

            with patch.object(WorkflowChatMessage, "get_or_none", new_callable=AsyncMock) as mock_get_msg:
                # No existing message linked to proposal
                mock_get_msg.return_value = None

                with patch.object(WorkflowChatMessage, "create", new_callable=AsyncMock) as mock_create:
                    mock_create.return_value = mock_chat_message

                    from seer.api.agents.workflow.services import save_chat_message

                    result = await save_chat_message(
                        session_id=1,
                        role="assistant",
                        content="Hello",
                        proposal=mock_proposal,
                    )

                    assert result == mock_chat_message
                    mock_get_msg.assert_called_once_with(proposal=mock_proposal)
                    mock_create.assert_called_once()
                    mock_chat_session.save.assert_called_once()


# =============================================================================
# Proposal Services Tests
# =============================================================================


@pytest.mark.unit
class TestProposalServices:
    """Tests for workflow proposal functions."""

    @pytest.mark.asyncio
    async def test_create_workflow_proposal(self, mock_workflow, mock_chat_session, mock_user, mock_proposal):
        """Test creating a workflow proposal."""
        from seer.database import WorkflowProposal

        valid_spec = {"version": "2", "nodes": [], "edges": []}

        with patch("seer.api.agents.workflow.services._normalize_spec") as mock_normalize:
            mock_normalize.return_value = valid_spec

            with patch.object(WorkflowProposal, "create", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_proposal

                from seer.api.agents.workflow.services import create_workflow_proposal

                result = await create_workflow_proposal(
                    workflow=mock_workflow,
                    session=mock_chat_session,
                    user=mock_user,
                    summary="Test proposal",
                    spec=valid_spec,
                )

                assert result == mock_proposal
                mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_workflow_proposal_truncates_long_summary(self, mock_workflow, mock_user, mock_proposal):
        """Test that long summaries are truncated."""
        from seer.database import WorkflowProposal

        valid_spec = {"version": "2", "nodes": [], "edges": []}
        long_summary = "x" * 600

        with patch("seer.api.agents.workflow.services._normalize_spec") as mock_normalize:
            mock_normalize.return_value = valid_spec

            with patch.object(WorkflowProposal, "create", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_proposal

                from seer.api.agents.workflow.services import create_workflow_proposal

                await create_workflow_proposal(
                    workflow=mock_workflow,
                    session=None,
                    user=mock_user,
                    summary=long_summary,
                    spec=valid_spec,
                )

                call_kwargs = mock_create.call_args.kwargs
                assert len(call_kwargs["summary"]) <= 512

    @pytest.mark.asyncio
    async def test_get_workflow_proposal_found(self, mock_workflow, mock_proposal):
        """Test getting an existing proposal."""
        from seer.database import WorkflowProposal

        with patch.object(WorkflowProposal, "get_or_none", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_proposal

            from seer.api.agents.workflow.services import get_workflow_proposal

            result = await get_workflow_proposal(workflow=mock_workflow, proposal_id=1)

            assert result == mock_proposal

    @pytest.mark.asyncio
    async def test_get_workflow_proposal_not_found(self, mock_workflow):
        """Test getting a non-existent proposal raises HTTPException."""
        from seer.database import WorkflowProposal

        with patch.object(WorkflowProposal, "get_or_none", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            from seer.api.agents.workflow.services import get_workflow_proposal

            with pytest.raises(HTTPException) as exc_info:
                await get_workflow_proposal(workflow=mock_workflow, proposal_id=999)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_accept_workflow_proposal(self, mock_workflow, mock_proposal, mock_user, mock_draft_version):
        """Test accepting a workflow proposal."""
        from seer.database import WorkflowProposal

        mock_proposal.status = WorkflowProposal.STATUS_PENDING
        mock_proposal.spec = {"version": "2", "nodes": [], "edges": []}

        # Create a coroutine that returns mock_workflow for the workflow FK
        async def get_workflow():
            return mock_workflow
        mock_proposal.workflow = get_workflow()

        with patch("seer.api.agents.workflow.services.get_workflow_proposal", new_callable=AsyncMock) as mock_get_prop:
            mock_get_prop.return_value = mock_proposal

            with patch("seer.api.agents.workflow.services._normalize_spec") as mock_normalize:
                mock_normalize.return_value = {"version": "2", "nodes": [], "edges": []}

                with patch("seer.api.agents.workflow.services._get_draft_version", new_callable=AsyncMock) as mock_get_draft:
                    mock_get_draft.return_value = mock_draft_version

                    with patch("seer.api.agents.workflow.services._update_draft_version", new_callable=AsyncMock):
                        from seer.api.agents.workflow.services import accept_workflow_proposal

                        result_proposal, _ = await accept_workflow_proposal(
                            workflow=mock_workflow,
                            proposal_id=1,
                            actor=mock_user,
                        )

                        assert result_proposal.status == WorkflowProposal.STATUS_ACCEPTED

    @pytest.mark.asyncio
    async def test_accept_workflow_proposal_not_pending(self, mock_workflow, mock_proposal, mock_user):
        """Test accepting a non-pending proposal raises HTTPException."""
        from seer.database import WorkflowProposal

        mock_proposal.status = WorkflowProposal.STATUS_ACCEPTED

        with patch("seer.api.agents.workflow.services.get_workflow_proposal", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_proposal

            from seer.api.agents.workflow.services import accept_workflow_proposal

            with pytest.raises(HTTPException) as exc_info:
                await accept_workflow_proposal(workflow=mock_workflow, proposal_id=1, actor=mock_user)

            assert exc_info.value.status_code == 400
            assert "not pending" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_accept_workflow_proposal_no_actor(self, mock_workflow, mock_proposal):
        """Test accepting a proposal without actor raises HTTPException."""
        from seer.database import WorkflowProposal

        mock_proposal.status = WorkflowProposal.STATUS_PENDING

        # Create a coroutine that returns mock_workflow for the workflow FK
        async def get_workflow():
            return mock_workflow
        mock_proposal.workflow = get_workflow()

        with patch("seer.api.agents.workflow.services.get_workflow_proposal", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_proposal

            from seer.api.agents.workflow.services import accept_workflow_proposal

            with pytest.raises(HTTPException) as exc_info:
                await accept_workflow_proposal(workflow=mock_workflow, proposal_id=1, actor=None)

            assert exc_info.value.status_code == 400
            assert "Actor is required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_reject_workflow_proposal(self, mock_workflow, mock_proposal):
        """Test rejecting a workflow proposal."""
        from seer.database import WorkflowProposal

        mock_proposal.status = WorkflowProposal.STATUS_PENDING

        with patch("seer.api.agents.workflow.services.get_workflow_proposal", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_proposal

            from seer.api.agents.workflow.services import reject_workflow_proposal

            result = await reject_workflow_proposal(workflow=mock_workflow, proposal_id=1)

            assert result.status == WorkflowProposal.STATUS_REJECTED
            mock_proposal.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_reject_workflow_proposal_not_pending(self, mock_workflow, mock_proposal):
        """Test rejecting a non-pending proposal raises HTTPException."""
        from seer.database import WorkflowProposal

        mock_proposal.status = WorkflowProposal.STATUS_REJECTED

        with patch("seer.api.agents.workflow.services.get_workflow_proposal", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_proposal

            from seer.api.agents.workflow.services import reject_workflow_proposal

            with pytest.raises(HTTPException) as exc_info:
                await reject_workflow_proposal(workflow=mock_workflow, proposal_id=1)

            assert exc_info.value.status_code == 400


# =============================================================================
# Preview From Spec Tests
# =============================================================================


@pytest.mark.unit
class TestPreviewFromSpec:
    """Tests for _preview_from_spec function."""

    def test_preview_from_spec_basic(self, sample_spec_dict):
        """Test building preview from spec."""
        from seer.api.agents.workflow.services import _preview_from_spec

        result = _preview_from_spec(sample_spec_dict)

        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 2
        assert result["nodes"][0]["id"] == "n1"
        assert result["nodes"][0]["type"] == "tool"

    def test_preview_from_spec_empty(self, minimal_spec_dict):
        """Test preview from empty spec."""
        from seer.api.agents.workflow.services import _preview_from_spec

        result = _preview_from_spec(minimal_spec_dict)

        assert result["nodes"] == []
        assert result["edges"] == []

    def test_preview_from_spec_generates_edges(self, sample_spec_dict):
        """Test that preview generates edges between sequential nodes."""
        from seer.api.agents.workflow.services import _preview_from_spec

        result = _preview_from_spec(sample_spec_dict)

        assert len(result["edges"]) == 1
        assert result["edges"][0]["source"] == "n1"
        assert result["edges"][0]["target"] == "n2"


# =============================================================================
# Discovery Session Services Tests
# =============================================================================


@pytest.mark.unit
class TestDiscoverySessionServices:
    """Tests for discovery chat session functions."""

    @pytest.mark.asyncio
    async def test_create_discovery_chat_session(self, mock_user):
        """Test creating a discovery chat session."""
        from seer.database.workflow_models import WorkflowDiscoveryChatSession, WorkflowCreationMode

        mock_session = MagicMock(spec=WorkflowDiscoveryChatSession)
        mock_session.id = 1
        mock_session.thread_id = "thread_123"
        mock_session.title = "New workflow"
        mock_session.workflow_creation_mode = WorkflowCreationMode.ASK_FIRST

        with patch.object(WorkflowDiscoveryChatSession, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_session

            from seer.api.agents.workflow.services import create_discovery_chat_session

            result = await create_discovery_chat_session(
                user=mock_user,
                thread_id="thread_123",
                workflow_creation_mode=WorkflowCreationMode.ASK_FIRST,
                title="New workflow",
            )

            assert result == mock_session
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_discovery_chat_session_found(self, mock_user):
        """Test getting an existing discovery session."""
        from seer.database.workflow_models import WorkflowDiscoveryChatSession

        mock_session = MagicMock(spec=WorkflowDiscoveryChatSession)
        mock_session.id = 1

        with patch.object(WorkflowDiscoveryChatSession, "get_or_none", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_session

            from seer.api.agents.workflow.services import get_discovery_chat_session

            result = await get_discovery_chat_session(session_id=1, user=mock_user)

            assert result == mock_session

    @pytest.mark.asyncio
    async def test_get_discovery_chat_session_not_found(self, mock_user):
        """Test getting non-existent discovery session raises HTTPException."""
        from seer.database.workflow_models import WorkflowDiscoveryChatSession

        with patch.object(WorkflowDiscoveryChatSession, "get_or_none", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            from seer.api.agents.workflow.services import get_discovery_chat_session

            with pytest.raises(HTTPException) as exc_info:
                await get_discovery_chat_session(session_id=999, user=mock_user)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_discovery_chat_session_by_thread_id(self, mock_user):
        """Test getting discovery session by thread ID."""
        from seer.database.workflow_models import WorkflowDiscoveryChatSession

        mock_session = MagicMock(spec=WorkflowDiscoveryChatSession)

        with patch.object(WorkflowDiscoveryChatSession, "filter") as mock_filter:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=mock_session)
            mock_filter.return_value = mock_query

            from seer.api.agents.workflow.services import get_discovery_chat_session_by_thread_id

            result = await get_discovery_chat_session_by_thread_id(thread_id="thread_123", user=mock_user)

            assert result == mock_session

    @pytest.mark.asyncio
    async def test_link_discovery_session_to_workflow(self, mock_workflow):
        """Test linking discovery session to workflow."""
        from seer.database.workflow_models import WorkflowDiscoveryChatSession

        mock_session = MagicMock(spec=WorkflowDiscoveryChatSession)
        mock_session.created_workflow = None
        mock_session.save = AsyncMock()

        from seer.api.agents.workflow.services import link_discovery_session_to_workflow

        await link_discovery_session_to_workflow(session=mock_session, workflow=mock_workflow)

        assert mock_session.created_workflow == mock_workflow
        mock_session.save.assert_called_once()


# =============================================================================
# User Workflow Creation Mode Tests
# =============================================================================


@pytest.mark.unit
class TestUserWorkflowCreationMode:
    """Tests for user workflow creation mode functions."""

    @pytest.mark.asyncio
    async def test_get_user_workflow_creation_mode_default(self, mock_user):
        """Test getting default workflow creation mode."""
        mock_user.default_workflow_creation_mode = None

        from seer.api.agents.workflow.services import get_user_workflow_creation_mode
        from seer.database.workflow_models import WorkflowCreationMode

        result = await get_user_workflow_creation_mode(mock_user)

        assert result == WorkflowCreationMode.ASK_FIRST

    @pytest.mark.asyncio
    async def test_get_user_workflow_creation_mode_custom(self, mock_user):
        """Test getting custom workflow creation mode."""
        mock_user.default_workflow_creation_mode = "AUTO_CREATE"

        from seer.api.agents.workflow.services import get_user_workflow_creation_mode
        from seer.database.workflow_models import WorkflowCreationMode

        result = await get_user_workflow_creation_mode(mock_user)

        assert result == WorkflowCreationMode.AUTO_CREATE

    @pytest.mark.asyncio
    async def test_update_user_workflow_creation_mode(self, mock_user):
        """Test updating user workflow creation mode."""
        from seer.api.agents.workflow.services import update_user_workflow_creation_mode
        from seer.database.workflow_models import WorkflowCreationMode

        result = await update_user_workflow_creation_mode(mock_user, WorkflowCreationMode.ON_ACCEPTANCE)

        assert mock_user.default_workflow_creation_mode == "ON_ACCEPTANCE"
        mock_user.save.assert_called_once()
        assert result == mock_user


# =============================================================================
# Normalize Spec Tests
# =============================================================================


@pytest.mark.unit
class TestNormalizeSpec:
    """Tests for _normalize_spec function."""

    def test_normalize_spec_none_input(self):
        """Test that None input raises HTTPException."""
        from seer.api.agents.workflow.services import _normalize_spec

        with pytest.raises(HTTPException) as exc_info:
            _normalize_spec(None)

        assert exc_info.value.status_code == 400
        assert "required" in exc_info.value.detail

    def test_normalize_spec_empty_dict(self):
        """Test that empty dict raises HTTPException."""
        from seer.api.agents.workflow.services import _normalize_spec

        with pytest.raises(HTTPException) as exc_info:
            _normalize_spec({})

        assert exc_info.value.status_code == 400

    def test_normalize_spec_valid(self, minimal_spec_dict):
        """Test normalizing a valid spec."""
        with patch("seer.core.compiler.parse.parse_workflow_spec") as mock_parse:
            mock_spec = MagicMock()
            mock_spec.model_dump.return_value = minimal_spec_dict
            mock_parse.return_value = mock_spec

            from seer.api.agents.workflow.services import _normalize_spec

            result = _normalize_spec(minimal_spec_dict)

            assert result == minimal_spec_dict
            mock_parse.assert_called_once_with(minimal_spec_dict)

    def test_normalize_spec_invalid(self):
        """Test that invalid spec raises HTTPException."""
        with patch("seer.core.compiler.parse.parse_workflow_spec") as mock_parse:
            mock_parse.side_effect = ValueError("Invalid spec")

            from seer.api.agents.workflow.services import _normalize_spec

            with pytest.raises(HTTPException) as exc_info:
                _normalize_spec({"invalid": "spec"})

            assert exc_info.value.status_code == 400
            assert "Invalid workflow spec" in exc_info.value.detail
