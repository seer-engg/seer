"""Unit tests for Google Docs tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from seer.tools.google.gdocs import (
    GoogleDocsReadTool,
    GoogleDocsWriteTool,
    GoogleDocsCreateTool,
)


@pytest.mark.unit
class TestGoogleDocsReadTool:
    """Tests for GoogleDocsReadTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = GoogleDocsReadTool()
        assert tool.name == "google_docs_read"
        assert tool.integration_type == "google_docs"
        assert "documents.readonly" in tool.required_scopes[0]

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = GoogleDocsReadTool()
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "document_id" in schema["properties"]
        assert "suggestions_view_mode" in schema["properties"]
        assert "document_id" in schema["required"]

    def test_get_output_schema(self):
        """Test output schema has document fields."""
        tool = GoogleDocsReadTool()
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "documentId" in schema["properties"]
        assert "title" in schema["properties"]
        assert "body" in schema["properties"]

    def test_get_resource_pickers(self):
        """Test resource pickers configuration."""
        tool = GoogleDocsReadTool()
        pickers = tool.get_resource_pickers()

        assert "document_id" in pickers
        assert pickers["document_id"]["resource_type"] == "google_document"

    @pytest.mark.asyncio
    async def test_execute_missing_document_id(self):
        """Test execute raises error when document_id is missing."""
        tool = GoogleDocsReadTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {})

        assert exc_info.value.status_code == 400
        assert "document_id is required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful document read."""
        tool = GoogleDocsReadTool()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "documentId": "doc123",
            "title": "Test Document",
            "body": {"content": []},
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute("test_token", {"document_id": "doc123"})

            assert result["documentId"] == "doc123"
            assert result["title"] == "Test Document"
            mock_request.assert_called_once()

            # Verify correct URL was called
            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"
            assert "docs.googleapis.com/v1/documents/doc123" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_execute_with_suggestions_view_mode(self):
        """Test execute with suggestions_view_mode parameter."""
        tool = GoogleDocsReadTool()

        mock_response = MagicMock()
        mock_response.json.return_value = {"documentId": "doc123"}

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            await tool.execute(
                "test_token",
                {
                    "document_id": "doc123",
                    "suggestions_view_mode": "PREVIEW_WITHOUT_SUGGESTIONS",
                },
            )

            call_args = mock_request.call_args
            params = call_args[1].get("params")
            assert params is not None
            assert params["suggestionsViewMode"] == "PREVIEW_WITHOUT_SUGGESTIONS"


@pytest.mark.unit
class TestGoogleDocsWriteTool:
    """Tests for GoogleDocsWriteTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = GoogleDocsWriteTool()
        assert tool.name == "google_docs_write"
        assert tool.integration_type == "google_docs"
        assert "documents" in tool.required_scopes[0]
        assert "readonly" not in tool.required_scopes[0]

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = GoogleDocsWriteTool()
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "document_id" in schema["properties"]
        assert "requests" in schema["properties"]
        assert "document_id" in schema["required"]
        assert "requests" in schema["required"]

    def test_get_output_schema(self):
        """Test output schema has batch update response fields."""
        tool = GoogleDocsWriteTool()
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "documentId" in schema["properties"]
        assert "replies" in schema["properties"]

    @pytest.mark.asyncio
    async def test_execute_missing_document_id(self):
        """Test execute raises error when document_id is missing."""
        tool = GoogleDocsWriteTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {"requests": []})

        assert exc_info.value.status_code == 400
        assert "document_id is required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_missing_requests(self):
        """Test execute raises error when requests is missing."""
        tool = GoogleDocsWriteTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {"document_id": "doc123"})

        assert exc_info.value.status_code == 400
        assert "requests is required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful document write."""
        tool = GoogleDocsWriteTool()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "documentId": "doc123",
            "replies": [{}],
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            requests = [{"insertText": {"text": "Hello", "location": {"index": 1}}}]
            result = await tool.execute(
                "test_token",
                {"document_id": "doc123", "requests": requests},
            )

            assert result["documentId"] == "doc123"
            mock_request.assert_called_once()

            # Verify correct URL and method
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert "docs.googleapis.com/v1/documents/doc123:batchUpdate" in call_args[0][1]

            # Verify request body
            json_body = call_args[1].get("json_body")
            assert json_body is not None
            assert json_body["requests"] == requests


@pytest.mark.unit
class TestGoogleDocsCreateTool:
    """Tests for GoogleDocsCreateTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = GoogleDocsCreateTool()
        assert tool.name == "google_docs_create"
        assert tool.integration_type == "google_docs"
        assert "documents" in tool.required_scopes[0]

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = GoogleDocsCreateTool()
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "title" in schema["properties"]
        assert "title" in schema["required"]

    def test_get_output_schema(self):
        """Test output schema has document fields."""
        tool = GoogleDocsCreateTool()
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "documentId" in schema["properties"]
        assert "title" in schema["properties"]

    def test_no_resource_pickers(self):
        """Test create tool has no resource pickers (creating new doc)."""
        tool = GoogleDocsCreateTool()
        # Create tool doesn't need pickers since it creates a new doc
        assert not hasattr(tool, "get_resource_pickers") or tool.get_resource_pickers() is None or tool.get_resource_pickers() == {}

    @pytest.mark.asyncio
    async def test_execute_missing_title(self):
        """Test execute raises error when title is missing."""
        tool = GoogleDocsCreateTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {})

        assert exc_info.value.status_code == 400
        assert "title is required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful document creation."""
        tool = GoogleDocsCreateTool()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "documentId": "new_doc_123",
            "title": "My New Document",
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                "test_token",
                {"title": "My New Document"},
            )

            assert result["documentId"] == "new_doc_123"
            assert result["title"] == "My New Document"
            mock_request.assert_called_once()

            # Verify correct URL and method
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert "docs.googleapis.com/v1/documents" in call_args[0][1]
            assert ":batchUpdate" not in call_args[0][1]

            # Verify request body
            json_body = call_args[1].get("json_body")
            assert json_body is not None
            assert json_body["title"] == "My New Document"
