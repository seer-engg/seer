"""
Mock tool implementations for testing.

Provides mock tools that can be used in tests without external dependencies
or API calls.
"""
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock


class MockTool:
    """
    Mock tool implementation for testing tool execution.

    Simulates a real tool with configurable behavior, allowing tests
    to verify tool execution logic without external dependencies.
    """

    def __init__(
        self,
        tool_id: str,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        execute_result: Optional[Dict[str, Any]] = None,
        execute_error: Optional[Exception] = None,
    ):
        """
        Initialize mock tool.

        Args:
            tool_id: Unique tool identifier (e.g., "test.mock_tool")
            name: Human-readable tool name
            description: Tool description
            parameters_schema: JSON schema for tool parameters
            execute_result: Result to return from execute() (if success)
            execute_error: Exception to raise from execute() (if failure)
        """
        self.id = tool_id
        self.name = name
        self.description = description
        self._parameters_schema = parameters_schema
        self._execute_result = execute_result or {"status": "success", "data": "mock_result"}
        self._execute_error = execute_error

        # Track execution history
        self.execution_count = 0
        self.last_parameters: Optional[Dict[str, Any]] = None

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return the tool's parameter schema."""
        return self._parameters_schema

    async def execute(self, parameters: Dict[str, Any], credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the mock tool.

        Args:
            parameters: Tool parameters
            credentials: Optional credentials

        Returns:
            Mock execution result

        Raises:
            Exception: If execute_error was provided
        """
        self.execution_count += 1
        self.last_parameters = parameters

        if self._execute_error:
            raise self._execute_error

        return self._execute_result

    def reset(self):
        """Reset execution history."""
        self.execution_count = 0
        self.last_parameters = None


def create_mock_tool(
    tool_id: str = "test.mock_tool",
    name: str = "Mock Tool",
    description: str = "A mock tool for testing",
    parameters: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Exception] = None,
) -> MockTool:
    """
    Factory function for creating mock tools with sensible defaults.

    Args:
        tool_id: Tool ID (default: "test.mock_tool")
        name: Tool name (default: "Mock Tool")
        description: Tool description
        parameters: Parameter schema properties (default: single string param)
        result: Execution result (default: success with mock data)
        error: Error to raise on execution

    Returns:
        Configured MockTool instance

    Example:
        # Simple mock tool
        tool = create_mock_tool()

        # Mock tool with custom result
        tool = create_mock_tool(
            tool_id="gmail.send_email",
            result={"message_id": "msg_123"}
        )

        # Mock tool that fails
        tool = create_mock_tool(
            tool_id="api.call",
            error=RuntimeError("API Error")
        )
    """
    # Default parameter schema
    if parameters is None:
        parameters = {
            "param1": {"type": "string", "description": "Test parameter"},
        }

    schema = {
        "type": "object",
        "properties": parameters,
        "required": list(parameters.keys()),
    }

    return MockTool(
        tool_id=tool_id,
        name=name,
        description=description,
        parameters_schema=schema,
        execute_result=result,
        execute_error=error,
    )


def create_gmail_send_email_mock() -> MockTool:
    """Create mock Gmail send email tool."""
    return create_mock_tool(
        tool_id="gmail.send_email",
        name="Send Email",
        description="Send an email via Gmail",
        parameters={
            "to": {"type": "string", "description": "Recipient email"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"},
        },
        result={
            "message_id": "msg_123",
            "status": "sent",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    )


def create_slack_send_message_mock() -> MockTool:
    """Create mock Slack send message tool."""
    return create_mock_tool(
        tool_id="slack.send_message",
        name="Send Message",
        description="Send a message to Slack",
        parameters={
            "channel": {"type": "string", "description": "Channel ID or name"},
            "text": {"type": "string", "description": "Message text"},
        },
        result={
            "message_id": "slack_msg_123",
            "channel": "general",
            "timestamp": "1609459200.000000",
        },
    )


def create_http_request_mock(response_data: Optional[Dict[str, Any]] = None) -> MockTool:
    """
    Create mock HTTP request tool.

    Args:
        response_data: Custom response data

    Returns:
        MockTool configured for HTTP requests
    """
    return create_mock_tool(
        tool_id="http.request",
        name="HTTP Request",
        description="Make an HTTP request",
        parameters={
            "url": {"type": "string", "description": "Request URL"},
            "method": {"type": "string", "description": "HTTP method"},
            "body": {"type": "object", "description": "Request body"},
        },
        result=response_data
        or {
            "status_code": 200,
            "body": {"result": "success"},
            "headers": {"content-type": "application/json"},
        },
    )


def create_failing_tool_mock(error_message: str = "Tool execution failed") -> MockTool:
    """
    Create a mock tool that always fails.

    Args:
        error_message: Error message to raise

    Returns:
        MockTool that raises RuntimeError on execution
    """
    return create_mock_tool(
        tool_id="test.failing_tool",
        name="Failing Tool",
        description="A tool that always fails",
        error=RuntimeError(error_message),
    )


def create_async_mock_tool() -> AsyncMock:
    """
    Create an AsyncMock configured as a tool.

    Useful for more complex mocking scenarios where you need
    full control over mock behavior.

    Returns:
        AsyncMock with tool interface
    """
    tool = AsyncMock()
    tool.id = "test.async_mock"
    tool.name = "Async Mock Tool"
    tool.description = "Async mock for testing"
    tool.get_parameters_schema.return_value = {
        "type": "object",
        "properties": {
            "param": {"type": "string"},
        },
    }
    tool.execute.return_value = {"status": "success"}
    return tool
