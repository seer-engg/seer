"""
Unit test fixtures.

Unit tests should be fast and isolated from external dependencies.
These fixtures provide mocks and helpers for testing pure functions and business logic.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Compiler Test Fixtures
# =============================================================================


@pytest.fixture
def mock_compiler():
    """
    Mock workflow compiler for testing compilation without full initialization.
    """
    compiler = MagicMock()
    compiler.compile.return_value = MagicMock()
    compiler.validate_spec.return_value = True
    return compiler


# =============================================================================
# Expression Evaluator Fixtures
# =============================================================================


@pytest.fixture
def eval_context():
    """
    Standard evaluation context for testing expressions.

    Provides common variables and helpers for expression evaluation tests.
    """
    return {
        "trigger": {
            "event_id": "evt_123",
            "data": {"message": "test", "value": 42},
        },
        "nodes": {
            "task_1": {
                "result": {"status": "success", "output": "hello"},
            },
        },
        "env": {
            "API_KEY": "test_key_123",
        },
    }


# =============================================================================
# Tool Registry Fixtures
# =============================================================================


@pytest.fixture
def mock_tool_registry():
    """
    Mock tool registry with a few test tools registered.
    """
    from seer.core.registry.tool_registry import ToolRegistry

    registry = ToolRegistry()

    # Register mock tools
    mock_tool_1 = MagicMock()
    mock_tool_1.id = "test.tool_1"
    mock_tool_1.name = "Test Tool 1"
    mock_tool_1.description = "A test tool"
    mock_tool_1.get_parameters_schema.return_value = {
        "type": "object",
        "properties": {"param": {"type": "string"}},
    }

    mock_tool_2 = MagicMock()
    mock_tool_2.id = "test.tool_2"
    mock_tool_2.name = "Test Tool 2"
    mock_tool_2.description = "Another test tool"
    mock_tool_2.get_parameters_schema.return_value = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
    }

    registry.register("test.tool_1", mock_tool_1)
    registry.register("test.tool_2", mock_tool_2)

    return registry


# =============================================================================
# Credit Calculator Fixtures
# =============================================================================


@pytest.fixture
def mock_usage_tracker():
    """
    Mock usage tracker for testing credit calculations without database.
    """
    tracker = AsyncMock()
    tracker.track_llm_usage.return_value = None
    tracker.get_current_usage.return_value = {
        "llm_credits_used": 0.0,
        "runs_this_month": 0,
    }
    return tracker


# =============================================================================
# Observability Fixtures
# =============================================================================


@pytest.fixture
def mock_limits():
    """
    Mock tier limits for testing usage limit checks.
    """
    from seer.observability.models import TierLimits

    return TierLimits(
        workflows=10,
        runs_monthly=1000,
        account_day_limit=30,
        poll_min_interval_seconds=60,
        llm_credits_monthly=10.0,
    )


# =============================================================================
# Common Mock Fixtures (Centralized to reduce duplication)
# =============================================================================


@pytest.fixture
def mock_user():
    """
    Centralized mock user with all common attributes.

    This fixture replaces duplicate mock_user fixtures across:
    - api/workflows/test_execution.py
    - api/workflows/test_triggers_service.py
    - api/agents/test_workflow_services.py
    - api/agents/test_chat_services.py
    - api/tools/test_services.py
    - api/integrations/test_services.py
    - api/core/test_usage_limit_middleware.py
    - tools/test_credential_resolver.py
    - agents/nexus/test_cost_callback.py
    - observability/test_cost_tracking.py
    """
    from seer.database import User

    user = MagicMock(spec=User)
    user.id = 1
    user.user_id = "user_123"
    user.email = "test@example.com"
    user.default_workflow_creation_mode = None
    user.save = AsyncMock()
    # For Tortoise ORM compatibility
    user._saved_in_db = True
    return user


@pytest.fixture
def mock_workflow():
    """
    Centralized mock workflow.

    This fixture replaces duplicate mock_workflow fixtures across:
    - api/workflows/test_execution.py
    - api/workflows/test_triggers_service.py
    - api/agents/test_workflow_services.py
    """
    from seer.database import Workflow

    workflow = MagicMock(spec=Workflow)
    workflow.id = 1
    workflow.workflow_id = "wf_1"
    workflow.name = "Test Workflow"
    return workflow


@pytest.fixture
def mock_workflow_version():
    """
    Centralized mock workflow version.

    This fixture replaces duplicate mock_workflow_version fixtures across:
    - api/workflows/test_execution.py
    - api/workflows/test_history.py
    """
    from seer.database import WorkflowVersion, WorkflowVersionStatus

    version = MagicMock(spec=WorkflowVersion)
    version.id = 1
    version.version_number = 1
    version.status = WorkflowVersionStatus.DRAFT
    version.spec = {"version": "2", "nodes": [], "edges": [], "triggers": []}
    version.save = AsyncMock()
    return version
