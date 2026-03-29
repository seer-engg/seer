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
    """Mock E2B SDK Result object.

    E2B SDK Results expose data via named attributes (.text, .json, .png, etc.)
    rather than a single .type field. This mock supports both the legacy (.type, .data)
    pattern and the E2B-style attribute pattern.
    """

    def __init__(
        self,
        result_type: str = "text",
        data: Any = "result_data",
        is_main_result: bool = True,
        *,
        text: Optional[str] = None,
        json: Optional[Any] = None,
        html: Optional[str] = None,
        markdown: Optional[str] = None,
        png: Optional[str] = None,
        jpeg: Optional[str] = None,
        svg: Optional[str] = None,
        pdf: Optional[str] = None,
        latex: Optional[str] = None,
        javascript: Optional[str] = None,
    ):
        self.type = result_type
        self.data = data
        self.is_main_result = is_main_result
        # E2B-style named attributes
        self.text = text
        self.json = json
        self.html = html
        self.markdown = markdown
        self.png = png
        self.jpeg = jpeg
        self.svg = svg
        self.pdf = pdf
        self.latex = latex
        self.javascript = javascript


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
    sandbox.files = MagicMock()
    sandbox.files.read = AsyncMock(return_value=b"mock file content")
    sandbox.files.write = AsyncMock()
    return sandbox


@pytest.fixture
def mock_async_sandbox_class(mock_sandbox):
    """Create a mock AsyncSandbox class with create/connect methods."""
    mock_class = MagicMock()
    mock_class.create = AsyncMock(return_value=mock_sandbox)
    mock_class.connect = AsyncMock(return_value=mock_sandbox)
    return mock_class
