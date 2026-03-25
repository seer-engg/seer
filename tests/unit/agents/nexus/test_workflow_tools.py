"""
Unit tests for submit_workflow_spec Nexus tool.

Tests the refactored submit_workflow_spec which delegates core validation
to the shared run_full_validation() golden function, keeping only
Nexus-specific pre-checks (thread context, spec format coercion, user resolution).
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from seer.tools.workflow_validation import ValidationResult, ValidationError


# Patch targets:
# - _current_thread_id and get_user_for_thread are module-level imports, so patch at consuming module.
# - run_full_validation is lazy-imported inside the function body, so patch at source module.
# - WorkflowChatSession and WorkflowProposal are lazy-imported inside the function body,
#   so patch at source module (seer.database.workflow_models).

_TOOLS_MOD = "seer.agents.nexus.tools.workflow_tools"
_VALIDATION_MOD = "seer.tools.workflow_validation"
_DB_MOD = "seer.database.workflow_models"


def _make_success_result(spec_dict, schema_fixes=None):
    """Helper to build a successful ValidationResult with a mock validated_spec."""
    mock_validated_spec = MagicMock()
    mock_validated_spec.model_dump.return_value = spec_dict
    return ValidationResult(
        success=True,
        validated_spec=mock_validated_spec,
        fixed_spec_dict=spec_dict,
        schema_fixes=schema_fixes or [],
    )


def _setup_session_and_proposal(mock_session_cls, mock_proposal_cls, proposal_id=42, session_found=True):
    """
    Helper to wire up WorkflowChatSession and WorkflowProposal mocks.

    The Tortoise ORM query ``WorkflowChatSession.get_or_none(...).prefetch_related(...)``
    is awaited, so ``prefetch_related`` must return a coroutine (AsyncMock).
    """
    mock_session = MagicMock() if session_found else None
    # get_or_none() returns a queryset; prefetch_related() is awaited
    mock_queryset = MagicMock()
    mock_queryset.prefetch_related = AsyncMock(return_value=mock_session)
    mock_session_cls.get_or_none.return_value = mock_queryset

    mock_proposal = MagicMock()
    mock_proposal.id = proposal_id
    mock_proposal_cls.create = AsyncMock(return_value=mock_proposal)
    mock_proposal_cls.STATUS_PENDING = "pending"
    return mock_session, mock_proposal


@pytest.mark.unit
class TestSubmitWorkflowSpec:
    """Tests for submit_workflow_spec Nexus agent tool."""

    @pytest.mark.asyncio
    async def test_missing_thread_context_returns_error(self):
        """Pre-check: missing thread_id context returns an internal error."""
        with patch(f"{_TOOLS_MOD}._current_thread_id") as mock_ctx:
            mock_ctx.get.return_value = None

            from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
            result = await submit_workflow_spec.ainvoke({"workflow_spec": {"version": "2", "nodes": []}})
            data = json.loads(result)

            assert data["status"] == "error"
            assert data["error_type"] == "internal"
            assert "thread_id" in data["message"]

    @pytest.mark.asyncio
    async def test_invalid_spec_format_returns_error(self):
        """Pre-check: non-dict/non-string spec returns a parsing error."""
        with patch(f"{_TOOLS_MOD}._current_thread_id") as mock_ctx:
            mock_ctx.get.return_value = "thread-123"

            from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
            result = await submit_workflow_spec.ainvoke({"workflow_spec": 12345})
            data = json.loads(result)

            assert data["status"] == "error"
            assert data["error_type"] == "parsing"

    @pytest.mark.asyncio
    async def test_json_string_spec_is_coerced(self):
        """Pre-check: JSON string spec is parsed into a dict before validation."""
        mock_user = MagicMock()
        spec_dict = {"version": "2", "nodes": []}

        with (
            patch(f"{_TOOLS_MOD}._current_thread_id") as mock_ctx,
            patch(f"{_TOOLS_MOD}.get_user_for_thread", new_callable=AsyncMock, return_value=mock_user),
            patch(f"{_VALIDATION_MOD}.run_full_validation", new_callable=AsyncMock) as mock_validate,
            patch(f"{_DB_MOD}.WorkflowChatSession") as mock_session_cls,
            patch(f"{_DB_MOD}.WorkflowProposal") as mock_proposal_cls,
        ):
            mock_ctx.get.return_value = "thread-123"
            mock_validate.return_value = _make_success_result(spec_dict)
            _setup_session_and_proposal(mock_session_cls, mock_proposal_cls)

            from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
            # Pass spec as JSON string
            result = await submit_workflow_spec.ainvoke({"workflow_spec": json.dumps(spec_dict)})
            data = json.loads(result)

            assert data["status"] == "ok"
            # Verify run_full_validation was called with the parsed dict
            mock_validate.assert_called_once_with(mock_user, spec_dict)

    @pytest.mark.asyncio
    async def test_user_not_found_returns_error(self):
        """User resolution failure returns an internal error."""
        with (
            patch(f"{_TOOLS_MOD}._current_thread_id") as mock_ctx,
            patch(f"{_TOOLS_MOD}.get_user_for_thread", new_callable=AsyncMock, return_value=None),
        ):
            mock_ctx.get.return_value = "thread-123"

            from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
            result = await submit_workflow_spec.ainvoke({"workflow_spec": {"version": "2", "nodes": []}})
            data = json.loads(result)

            assert data["status"] == "error"
            assert data["error_type"] == "internal"
            assert "User context" in data["message"]

    @pytest.mark.asyncio
    async def test_validation_failure_propagates_error(self):
        """run_full_validation failure is propagated with error_type, message, and hint."""
        mock_user = MagicMock()

        with (
            patch(f"{_TOOLS_MOD}._current_thread_id") as mock_ctx,
            patch(f"{_TOOLS_MOD}.get_user_for_thread", new_callable=AsyncMock, return_value=mock_user),
            patch(f"{_VALIDATION_MOD}.run_full_validation", new_callable=AsyncMock) as mock_validate,
        ):
            mock_ctx.get.return_value = "thread-123"
            mock_validate.return_value = ValidationResult(
                success=False,
                error=ValidationError(
                    "schema_validation",
                    "Invalid spec: missing required field 'version'",
                    "Check that your spec follows the workflow schema",
                ),
            )

            from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
            result = await submit_workflow_spec.ainvoke({"workflow_spec": {"nodes": []}})
            data = json.loads(result)

            assert data["status"] == "error"
            assert data["error_type"] == "schema_validation"
            assert "missing required field" in data["message"]
            assert "hint" in data

    @pytest.mark.asyncio
    async def test_validation_failure_without_hint(self):
        """run_full_validation failure without a hint omits the hint field."""
        mock_user = MagicMock()

        with (
            patch(f"{_TOOLS_MOD}._current_thread_id") as mock_ctx,
            patch(f"{_TOOLS_MOD}.get_user_for_thread", new_callable=AsyncMock, return_value=mock_user),
            patch(f"{_VALIDATION_MOD}.run_full_validation", new_callable=AsyncMock) as mock_validate,
        ):
            mock_ctx.get.return_value = "thread-123"
            mock_validate.return_value = ValidationResult(
                success=False,
                error=ValidationError("compilation", "Compilation failed: graph cycle detected"),
            )

            from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
            result = await submit_workflow_spec.ainvoke({"workflow_spec": {"version": "2", "nodes": []}})
            data = json.loads(result)

            assert data["status"] == "error"
            assert data["error_type"] == "compilation"
            # hint is always present now (includes common fix advice)
            assert "Common fix" in data["hint"]

    @pytest.mark.asyncio
    async def test_success_creates_proposal(self):
        """Successful validation creates WorkflowProposal and returns spec + proposal_id."""
        mock_user = MagicMock()
        spec_dict = {"version": "2", "nodes": []}

        with (
            patch(f"{_TOOLS_MOD}._current_thread_id") as mock_ctx,
            patch(f"{_TOOLS_MOD}.get_user_for_thread", new_callable=AsyncMock, return_value=mock_user),
            patch(f"{_VALIDATION_MOD}.run_full_validation", new_callable=AsyncMock) as mock_validate,
            patch(f"{_DB_MOD}.WorkflowChatSession") as mock_session_cls,
            patch(f"{_DB_MOD}.WorkflowProposal") as mock_proposal_cls,
        ):
            mock_ctx.get.return_value = "thread-123"
            mock_validate.return_value = _make_success_result(spec_dict)
            _setup_session_and_proposal(mock_session_cls, mock_proposal_cls, proposal_id=42)

            from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
            result = await submit_workflow_spec.ainvoke({"workflow_spec": spec_dict})
            data = json.loads(result)

            assert data["status"] == "ok"
            assert data["proposal_id"] == 42
            assert data["workflow_spec"] == spec_dict
            assert "auto_fixes" not in data

    @pytest.mark.asyncio
    async def test_success_with_auto_fixes(self):
        """Schema fixes from run_full_validation are included in the response."""
        mock_user = MagicMock()
        spec_dict = {"version": "2", "nodes": []}
        schema_fixes = [
            {"trigger_key": "gmail_new_email", "field": "event_schema", "action": "replaced_with_canonical"}
        ]

        with (
            patch(f"{_TOOLS_MOD}._current_thread_id") as mock_ctx,
            patch(f"{_TOOLS_MOD}.get_user_for_thread", new_callable=AsyncMock, return_value=mock_user),
            patch(f"{_VALIDATION_MOD}.run_full_validation", new_callable=AsyncMock) as mock_validate,
            patch(f"{_DB_MOD}.WorkflowChatSession") as mock_session_cls,
            patch(f"{_DB_MOD}.WorkflowProposal") as mock_proposal_cls,
        ):
            mock_ctx.get.return_value = "thread-123"
            mock_validate.return_value = _make_success_result(spec_dict, schema_fixes=schema_fixes)
            _setup_session_and_proposal(mock_session_cls, mock_proposal_cls, proposal_id=99)

            from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
            result = await submit_workflow_spec.ainvoke({"workflow_spec": spec_dict})
            data = json.loads(result)

            assert data["status"] == "ok"
            assert "auto_fixes" in data
            assert data["auto_fixes"]["trigger_schemas_updated"] == schema_fixes

    @pytest.mark.asyncio
    async def test_success_with_custom_summary(self):
        """Custom summary is included in the response and passed to the proposal."""
        mock_user = MagicMock()
        spec_dict = {"version": "2", "nodes": []}

        with (
            patch(f"{_TOOLS_MOD}._current_thread_id") as mock_ctx,
            patch(f"{_TOOLS_MOD}.get_user_for_thread", new_callable=AsyncMock, return_value=mock_user),
            patch(f"{_VALIDATION_MOD}.run_full_validation", new_callable=AsyncMock) as mock_validate,
            patch(f"{_DB_MOD}.WorkflowChatSession") as mock_session_cls,
            patch(f"{_DB_MOD}.WorkflowProposal") as mock_proposal_cls,
        ):
            mock_ctx.get.return_value = "thread-123"
            mock_validate.return_value = _make_success_result(spec_dict)
            _setup_session_and_proposal(mock_session_cls, mock_proposal_cls, proposal_id=7)

            from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
            result = await submit_workflow_spec.ainvoke({
                "workflow_spec": spec_dict,
                "summary": "Send welcome email on signup",
            })
            data = json.loads(result)

            assert data["status"] == "ok"
            assert data["summary"] == "Send welcome email on signup"
            # Verify summary was passed to proposal creation
            create_kwargs = mock_proposal_cls.create.call_args[1]
            assert create_kwargs["summary"] == "Send welcome email on signup"

    @pytest.mark.asyncio
    async def test_session_not_found_returns_error(self):
        """Missing chat session returns an internal error."""
        mock_user = MagicMock()
        spec_dict = {"version": "2", "nodes": []}

        with (
            patch(f"{_TOOLS_MOD}._current_thread_id") as mock_ctx,
            patch(f"{_TOOLS_MOD}.get_user_for_thread", new_callable=AsyncMock, return_value=mock_user),
            patch(f"{_VALIDATION_MOD}.run_full_validation", new_callable=AsyncMock) as mock_validate,
            patch(f"{_DB_MOD}.WorkflowChatSession") as mock_session_cls,
        ):
            mock_ctx.get.return_value = "thread-123"
            mock_validate.return_value = _make_success_result(spec_dict)
            _setup_session_and_proposal(mock_session_cls, MagicMock(), session_found=False)

            from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
            result = await submit_workflow_spec.ainvoke({"workflow_spec": spec_dict})
            data = json.loads(result)

            assert data["status"] == "error"
            assert data["error_type"] == "internal"
            assert "Chat session" in data["message"]
