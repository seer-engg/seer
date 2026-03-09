"""
Unit tests for Notion tools.

Tests cover:
- NotionSearchTool
- NotionGetPageTool
- NotionCreatePageTool
- NotionUpdatePageTool
- NotionGetPageContentTool
- NotionAppendPageContentTool
- NotionQueryDatabaseTool
- NotionCreateDatabaseEntryTool

All HTTP calls are mocked using pytest-mock and respx/unittest.mock.
"""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx

from fastapi import HTTPException

from seer.tools.notion.search import NotionSearchTool
from seer.tools.notion.pages import (
    NotionGetPageTool,
    NotionCreatePageTool,
    NotionUpdatePageTool,
    NotionGetPageContentTool,
    NotionAppendPageContentTool,
    _extract_page_title,
    _extract_block_text,
)
from seer.tools.notion.databases import (
    NotionQueryDatabaseTool,
    NotionCreateDatabaseEntryTool,
    _extract_entry_title,
    _format_property_value,
)
from seer.tools.notion.base import NotionAPIClient, NOTION_VERSION
from seer.tools.credential_resolver import ResolvedCredentials


def make_credentials(token: str = "test-access-token") -> ResolvedCredentials:
    """Create a mock ResolvedCredentials with an access token."""
    creds = MagicMock(spec=ResolvedCredentials)
    creds.access_token = token
    return creds


# ─── Unit Helpers ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestNotionPageHelpers:
    """Test page helper functions."""

    def test_extract_page_title_from_title_property(self):
        page = {
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "Hello World"}],
                }
            }
        }
        assert _extract_page_title(page) == "Hello World"

    def test_extract_page_title_concatenates_multiple_parts(self):
        page = {
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [
                        {"plain_text": "Hello"},
                        {"plain_text": " World"},
                    ],
                }
            }
        }
        assert _extract_page_title(page) == "Hello World"

    def test_extract_page_title_no_title_property_returns_untitled(self):
        page = {"properties": {"Status": {"type": "select"}}}
        assert _extract_page_title(page) == "(Untitled)"

    def test_extract_block_text_paragraph(self):
        block = {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"plain_text": "Some text"},
                    {"plain_text": " more text"},
                ]
            },
        }
        assert _extract_block_text(block) == "Some text more text"

    def test_extract_block_text_empty_rich_text(self):
        block = {"type": "heading_1", "heading_1": {"rich_text": []}}
        assert _extract_block_text(block) == ""


@pytest.mark.unit
class TestNotionDatabaseHelpers:
    """Test database helper functions."""

    def test_extract_entry_title_from_title_property(self):
        entry = {
            "properties": {
                "Task": {
                    "type": "title",
                    "title": [{"plain_text": "Fix bug"}],
                }
            }
        }
        assert _extract_entry_title(entry) == "Fix bug"

    def test_format_property_value_title(self):
        prop = {"type": "title", "title": [{"plain_text": "My Page"}]}
        assert _format_property_value(prop) == "My Page"

    def test_format_property_value_number(self):
        prop = {"type": "number", "number": 42}
        assert _format_property_value(prop) == 42

    def test_format_property_value_select(self):
        prop = {"type": "select", "select": {"name": "In Progress"}}
        assert _format_property_value(prop) == "In Progress"

    def test_format_property_value_multi_select(self):
        prop = {
            "type": "multi_select",
            "multi_select": [{"name": "Tag A"}, {"name": "Tag B"}],
        }
        assert _format_property_value(prop) == ["Tag A", "Tag B"]

    def test_format_property_value_checkbox(self):
        prop = {"type": "checkbox", "checkbox": True}
        assert _format_property_value(prop) is True

    def test_format_property_value_url(self):
        prop = {"type": "url", "url": "https://example.com"}
        assert _format_property_value(prop) == "https://example.com"

    def test_format_property_value_status(self):
        prop = {"type": "status", "status": {"name": "Done"}}
        assert _format_property_value(prop) == "Done"

    def test_format_property_value_null_select(self):
        prop = {"type": "select", "select": None}
        assert _format_property_value(prop) is None


# ─── NotionAPIClient base ────────────────────────────────────────────────────

@pytest.mark.unit
class TestNotionAPIClient:
    """Test NotionAPIClient base class functionality."""

    def test_get_access_token_returns_token_from_credentials(self):
        tool = NotionSearchTool()
        creds = make_credentials("my-token")
        assert tool._get_access_token(creds) == "my-token"

    def test_get_access_token_raises_401_without_credentials(self):
        tool = NotionSearchTool()
        with pytest.raises(HTTPException) as exc_info:
            tool._get_access_token(None)
        assert exc_info.value.status_code == 401

    def test_build_headers_sets_notion_version(self):
        tool = NotionSearchTool()
        headers = tool._build_headers("test-token")
        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Notion-Version"] == NOTION_VERSION
        assert headers["Content-Type"] == "application/json"

    def test_handle_notion_error_unauthorized(self):
        tool = NotionSearchTool()
        response = MagicMock()
        response.status_code = 401
        response.json.return_value = {"code": "unauthorized", "message": "Unauthorized"}
        exc = httpx.HTTPStatusError("401", request=MagicMock(), response=response)
        result = tool._handle_notion_error(exc)
        assert result.status_code == 401
        assert "reconnect" in result.detail.lower()

    def test_handle_notion_error_object_not_found(self):
        tool = NotionSearchTool()
        response = MagicMock()
        response.status_code = 404
        response.json.return_value = {"code": "object_not_found", "message": "Not found"}
        exc = httpx.HTTPStatusError("404", request=MagicMock(), response=response)
        result = tool._handle_notion_error(exc)
        assert result.status_code == 404

    def test_handle_notion_error_rate_limited(self):
        tool = NotionSearchTool()
        response = MagicMock()
        response.status_code = 429
        response.json.return_value = {"code": "rate_limited", "message": "Too many requests"}
        exc = httpx.HTTPStatusError("429", request=MagicMock(), response=response)
        result = tool._handle_notion_error(exc)
        assert result.status_code == 429

    def test_handle_notion_error_validation_error(self):
        tool = NotionSearchTool()
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {"code": "validation_error", "message": "Invalid param"}
        exc = httpx.HTTPStatusError("400", request=MagicMock(), response=response)
        result = tool._handle_notion_error(exc)
        assert result.status_code == 400

    def test_handle_notion_error_json_parse_failure(self):
        tool = NotionSearchTool()
        response = MagicMock()
        response.status_code = 500
        response.json.side_effect = ValueError("Not JSON")
        response.text = "Internal error"
        exc = httpx.HTTPStatusError("500", request=MagicMock(), response=response)
        result = tool._handle_notion_error(exc)
        assert result.status_code == 500


# ─── NotionSearchTool ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestNotionSearchTool:
    """Test NotionSearchTool."""

    def test_tool_attributes(self):
        tool = NotionSearchTool()
        assert tool.name == "notion_search"
        assert tool.provider == "notion"
        assert tool.integration_type == "notion"

    def test_schema_has_required_query(self):
        tool = NotionSearchTool()
        schema = tool.get_parameters_schema()
        assert "query" in schema["required"]

    @pytest.mark.asyncio
    async def test_execute_raises_if_no_query(self):
        tool = NotionSearchTool()
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {}, credentials=make_credentials())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_returns_results(self):
        tool = NotionSearchTool()
        mock_response = {
            "results": [
                {
                    "id": "page-123",
                    "object": "page",
                    "url": "https://notion.so/page-123",
                    "last_edited_time": "2024-01-01T00:00:00.000Z",
                    "properties": {
                        "title": {
                            "type": "title",
                            "title": [{"plain_text": "My Page"}],
                        }
                    },
                }
            ],
            "has_more": False,
        }
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)):
            result = await tool.execute(
                None, {"query": "My Page"}, credentials=make_credentials()
            )

        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "page-123"
        assert result["results"][0]["title"] == "My Page"
        assert result["results"][0]["type"] == "page"
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_execute_passes_filter_type(self):
        tool = NotionSearchTool()
        mock_response = {"results": [], "has_more": False}
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None,
                {"query": "test", "filter_type": "database"},
                credentials=make_credentials(),
            )
        called_body = mock_req.call_args[1]["json_body"]
        assert called_body["filter"] == {"value": "database", "property": "object"}

    @pytest.mark.asyncio
    async def test_execute_respects_page_size_cap(self):
        tool = NotionSearchTool()
        mock_response = {"results": [], "has_more": False}
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None,
                {"query": "test", "page_size": 9999},
                credentials=make_credentials(),
            )
        called_body = mock_req.call_args[1]["json_body"]
        assert called_body["page_size"] == 100


# ─── NotionGetPageTool ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestNotionGetPageTool:
    """Test NotionGetPageTool."""

    def test_tool_attributes(self):
        tool = NotionGetPageTool()
        assert tool.name == "notion_get_page"
        assert tool.provider == "notion"

    @pytest.mark.asyncio
    async def test_execute_raises_if_no_page_id(self):
        tool = NotionGetPageTool()
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {}, credentials=make_credentials())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_returns_page_data(self):
        tool = NotionGetPageTool()
        mock_page = {
            "id": "abc-123",
            "url": "https://notion.so/abc-123",
            "created_time": "2024-01-01T00:00:00.000Z",
            "last_edited_time": "2024-01-02T00:00:00.000Z",
            "archived": False,
            "properties": {
                "title": {
                    "type": "title",
                    "title": [{"plain_text": "Test Page"}],
                }
            },
        }
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_page)):
            result = await tool.execute(
                None, {"page_id": "abc-123"}, credentials=make_credentials()
            )

        assert result["id"] == "abc-123"
        assert result["title"] == "Test Page"
        assert result["archived"] is False
        assert result["url"] == "https://notion.so/abc-123"


# ─── NotionCreatePageTool ─────────────────────────────────────────────────────

@pytest.mark.unit
class TestNotionCreatePageTool:
    """Test NotionCreatePageTool."""

    @pytest.mark.asyncio
    async def test_execute_raises_if_missing_required_params(self):
        tool = NotionCreatePageTool()
        with pytest.raises(HTTPException):
            await tool.execute(None, {"parent_id": "p1"}, credentials=make_credentials())

    @pytest.mark.asyncio
    async def test_execute_creates_page_in_page(self):
        tool = NotionCreatePageTool()
        mock_response = {"id": "new-page-id", "url": "https://notion.so/new-page-id"}
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            result = await tool.execute(
                None,
                {"parent_id": "parent-id", "parent_type": "page", "title": "New Page"},
                credentials=make_credentials(),
            )

        assert result["id"] == "new-page-id"
        assert result["url"] == "https://notion.so/new-page-id"
        called_body = mock_req.call_args[1]["json_body"]
        assert called_body["parent"] == {"page_id": "parent-id"}

    @pytest.mark.asyncio
    async def test_execute_creates_page_in_database(self):
        tool = NotionCreatePageTool()
        mock_response = {"id": "entry-id", "url": "https://notion.so/entry-id"}
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None,
                {"parent_id": "db-id", "parent_type": "database", "title": "New Entry"},
                credentials=make_credentials(),
            )

        called_body = mock_req.call_args[1]["json_body"]
        assert called_body["parent"] == {"database_id": "db-id"}

    @pytest.mark.asyncio
    async def test_execute_includes_content_block(self):
        tool = NotionCreatePageTool()
        mock_response = {"id": "id", "url": "https://notion.so/id"}
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None,
                {
                    "parent_id": "parent",
                    "parent_type": "page",
                    "title": "My Page",
                    "content": "Hello Notion",
                },
                credentials=make_credentials(),
            )

        called_body = mock_req.call_args[1]["json_body"]
        assert "children" in called_body
        assert called_body["children"][0]["type"] == "paragraph"


# ─── NotionUpdatePageTool ─────────────────────────────────────────────────────

@pytest.mark.unit
class TestNotionUpdatePageTool:
    """Test NotionUpdatePageTool."""

    @pytest.mark.asyncio
    async def test_execute_raises_if_no_page_id(self):
        tool = NotionUpdatePageTool()
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {}, credentials=make_credentials())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_raises_if_no_update_fields(self):
        tool = NotionUpdatePageTool()
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {"page_id": "p1"}, credentials=make_credentials())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_updates_title(self):
        tool = NotionUpdatePageTool()
        mock_response = {"id": "p1", "url": "https://notion.so/p1"}
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            result = await tool.execute(
                None,
                {"page_id": "p1", "title": "Updated Title"},
                credentials=make_credentials(),
            )

        assert result["id"] == "p1"
        called_body = mock_req.call_args[1]["json_body"]
        assert "properties" in called_body

    @pytest.mark.asyncio
    async def test_execute_archives_page(self):
        tool = NotionUpdatePageTool()
        mock_response = {"id": "p1", "url": "https://notion.so/p1"}
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None,
                {"page_id": "p1", "archived": True},
                credentials=make_credentials(),
            )

        called_body = mock_req.call_args[1]["json_body"]
        assert called_body["archived"] is True


# ─── NotionGetPageContentTool ─────────────────────────────────────────────────

@pytest.mark.unit
class TestNotionGetPageContentTool:
    """Test NotionGetPageContentTool."""

    @pytest.mark.asyncio
    async def test_execute_raises_if_no_page_id(self):
        tool = NotionGetPageContentTool()
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {}, credentials=make_credentials())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_returns_blocks(self):
        tool = NotionGetPageContentTool()
        mock_response = {
            "results": [
                {
                    "id": "block-1",
                    "type": "paragraph",
                    "has_children": False,
                    "paragraph": {
                        "rich_text": [{"plain_text": "Hello world"}]
                    },
                }
            ],
            "has_more": False,
        }
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)):
            result = await tool.execute(
                None, {"page_id": "p1"}, credentials=make_credentials()
            )

        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["id"] == "block-1"
        assert result["blocks"][0]["type"] == "paragraph"
        assert result["blocks"][0]["text"] == "Hello world"
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_execute_caps_page_size_at_100(self):
        tool = NotionGetPageContentTool()
        mock_response = {"results": [], "has_more": False}
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None, {"page_id": "p1", "page_size": 9999}, credentials=make_credentials()
            )

        called_params = mock_req.call_args[1]["params"]
        assert called_params["page_size"] == 100


# ─── NotionAppendPageContentTool ──────────────────────────────────────────────

@pytest.mark.unit
class TestNotionAppendPageContentTool:
    """Test NotionAppendPageContentTool."""

    @pytest.mark.asyncio
    async def test_execute_raises_if_missing_params(self):
        tool = NotionAppendPageContentTool()
        with pytest.raises(HTTPException):
            await tool.execute(None, {"page_id": "p1"}, credentials=make_credentials())

    @pytest.mark.asyncio
    async def test_execute_appends_paragraph_block(self):
        tool = NotionAppendPageContentTool()
        mock_response = {
            "results": [{"id": "new-block-id"}]
        }
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            result = await tool.execute(
                None,
                {"page_id": "p1", "text": "Appended text"},
                credentials=make_credentials(),
            )

        assert result["block_ids"] == ["new-block-id"]
        called_body = mock_req.call_args[1]["json_body"]
        assert called_body["children"][0]["type"] == "paragraph"
        assert called_body["children"][0]["paragraph"]["rich_text"][0]["text"]["content"] == "Appended text"


# ─── NotionQueryDatabaseTool ──────────────────────────────────────────────────

@pytest.mark.unit
class TestNotionQueryDatabaseTool:
    """Test NotionQueryDatabaseTool."""

    def test_tool_attributes(self):
        tool = NotionQueryDatabaseTool()
        assert tool.name == "notion_query_database"
        assert tool.provider == "notion"

    @pytest.mark.asyncio
    async def test_execute_raises_if_no_database_id(self):
        tool = NotionQueryDatabaseTool()
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {}, credentials=make_credentials())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_returns_entries(self):
        tool = NotionQueryDatabaseTool()
        mock_response = {
            "results": [
                {
                    "id": "entry-1",
                    "url": "https://notion.so/entry-1",
                    "created_time": "2024-01-01T00:00:00.000Z",
                    "last_edited_time": "2024-01-02T00:00:00.000Z",
                    "properties": {
                        "Name": {
                            "type": "title",
                            "title": [{"plain_text": "Task One"}],
                        },
                        "Status": {
                            "type": "select",
                            "select": {"name": "Done"},
                        },
                    },
                }
            ],
            "has_more": False,
        }
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)):
            result = await tool.execute(
                None, {"database_id": "db-1"}, credentials=make_credentials()
            )

        assert len(result["entries"]) == 1
        entry = result["entries"][0]
        assert entry["id"] == "entry-1"
        assert entry["title"] == "Task One"
        assert entry["properties"]["Status"] == "Done"
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_execute_passes_filter_and_sorts(self):
        tool = NotionQueryDatabaseTool()
        mock_response = {"results": [], "has_more": False}
        filter_obj = {"property": "Status", "select": {"equals": "Done"}}
        sorts = [{"property": "Name", "direction": "ascending"}]

        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None,
                {"database_id": "db-1", "filter": filter_obj, "sorts": sorts},
                credentials=make_credentials(),
            )

        called_body = mock_req.call_args[1]["json_body"]
        assert called_body["filter"] == filter_obj
        assert called_body["sorts"] == sorts

    @pytest.mark.asyncio
    async def test_execute_caps_page_size(self):
        tool = NotionQueryDatabaseTool()
        mock_response = {"results": [], "has_more": False}
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None,
                {"database_id": "db-1", "page_size": 9999},
                credentials=make_credentials(),
            )
        called_body = mock_req.call_args[1]["json_body"]
        assert called_body["page_size"] == 100


# ─── NotionCreateDatabaseEntryTool ────────────────────────────────────────────

@pytest.mark.unit
class TestNotionCreateDatabaseEntryTool:
    """Test NotionCreateDatabaseEntryTool."""

    @pytest.mark.asyncio
    async def test_execute_raises_if_no_database_id(self):
        tool = NotionCreateDatabaseEntryTool()
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                None, {"properties": {"Name": {}}}, credentials=make_credentials()
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_raises_if_no_properties(self):
        tool = NotionCreateDatabaseEntryTool()
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                None, {"database_id": "db-1"}, credentials=make_credentials()
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_creates_entry(self):
        tool = NotionCreateDatabaseEntryTool()
        mock_response = {"id": "new-entry", "url": "https://notion.so/new-entry"}
        props = {
            "Name": {"title": [{"text": {"content": "New Task"}}]},
            "Status": {"select": {"name": "In Progress"}},
        }
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            result = await tool.execute(
                None,
                {"database_id": "db-1", "properties": props},
                credentials=make_credentials(),
            )

        assert result["id"] == "new-entry"
        assert result["url"] == "https://notion.so/new-entry"
        called_body = mock_req.call_args[1]["json_body"]
        assert called_body["parent"] == {"database_id": "db-1"}
        assert called_body["properties"] == props
