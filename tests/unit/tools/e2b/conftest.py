"""Test fixtures for E2B sandbox tools."""
from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockExecutionError:
    """Mock execution error."""

    def __init__(self, name: str, value: str, traceback: str):
        self.name = name
        self.value = value
        self.traceback = traceback

    def __str__(self) -> str:
        return f"{self.name}: {self.value}"


class MockResult:
    """Mock execution result."""

    def __init__(
        self,
        result_type: str = "text",
        data: Any = "result_data",
        is_main_result: bool = True,
    ):
        self.type = result_type
        self.data = data
        self.is_main_result = is_main_result


class MockLogs:
    """Mock execution logs."""

    def __init__(self, stdout: str = "", stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr


class MockExecution:
    """Mock code execution result."""

    def __init__(
        self,
        results: Optional[List[MockResult]] = None,
        logs: Optional[MockLogs] = None,
        error: Optional[MockExecutionError] = None,
    ):
        self.results = results or []
        self.logs = logs or MockLogs()
        self.error = error


@pytest.fixture
def mock_sandbox():
    """Create a mock AsyncSandbox."""
    sandbox = AsyncMock()
    sandbox.sandbox_id = "sbx_test123"
    sandbox.run_code = AsyncMock(return_value=MockExecution())
    sandbox.kill = AsyncMock()
    return sandbox


@pytest.fixture
def mock_async_sandbox_class(mock_sandbox):
    """Create a mock AsyncSandbox class with create/connect methods."""
    mock_class = MagicMock()
    mock_class.create = AsyncMock(return_value=mock_sandbox)
    mock_class.connect = AsyncMock(return_value=mock_sandbox)
    return mock_class
