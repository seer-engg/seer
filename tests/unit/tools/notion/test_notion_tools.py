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
from seer.tools.notion.markdown_blocks import (
    _build_paragraph_blocks,
    markdown_to_notion_blocks as _markdown_to_notion_blocks,
    _NOTION_MAX_TEXT_LENGTH,
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


# ─── _build_paragraph_blocks chunking ────────────────────────────────────────

@pytest.mark.unit
class TestBuildParagraphBlocksChunking:
    """Verify _build_paragraph_blocks splits text at the 2000-char Notion limit."""

    def test_short_text_returns_single_block(self):
        blocks = _build_paragraph_blocks("hello")
        assert len(blocks) == 1
        assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "hello"

    def test_exactly_2000_chars_returns_single_block(self):
        text = "a" * _NOTION_MAX_TEXT_LENGTH
        blocks = _build_paragraph_blocks(text)
        assert len(blocks) == 1
        assert len(blocks[0]["paragraph"]["rich_text"][0]["text"]["content"]) == 2000

    def test_text_over_2000_chars_is_chunked(self):
        """Regression test for the 2770-char incident on 2026-03-30."""
        text = "x" * 2770
        blocks = _build_paragraph_blocks(text)
        assert len(blocks) == 2
        assert len(blocks[0]["paragraph"]["rich_text"][0]["text"]["content"]) == 2000
        assert len(blocks[1]["paragraph"]["rich_text"][0]["text"]["content"]) == 770

    def test_large_text_chunked_into_many_blocks(self):
        text = "b" * 6100
        blocks = _build_paragraph_blocks(text)
        assert len(blocks) == 4  # 2000 + 2000 + 2000 + 100
        assert len(blocks[3]["paragraph"]["rich_text"][0]["text"]["content"]) == 100

    def test_each_block_has_correct_structure(self):
        blocks = _build_paragraph_blocks("test content")
        block = blocks[0]
        assert block["object"] == "block"
        assert block["type"] == "paragraph"
        assert block["paragraph"]["rich_text"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_create_page_with_long_content_sends_multiple_children(self):
        """End-to-end regression: NotionCreatePageTool must chunk content > 2000 chars."""
        tool = NotionCreatePageTool()
        mock_response = {"id": "id", "url": "https://notion.so/id"}
        long_content = "z" * 2770
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None,
                {
                    "parent_id": "parent",
                    "parent_type": "page",
                    "title": "My Page",
                    "content": long_content,
                },
                credentials=make_credentials(),
            )

        called_body = mock_req.call_args[1]["json_body"]
        # Markdown converter wraps plain text in a single <p>, chunking happens inside rich_text
        assert len(called_body["children"]) == 1
        rich_text = called_body["children"][0]["paragraph"]["rich_text"]
        assert len(rich_text) == 2
        assert len(rich_text[0]["text"]["content"]) == 2000
        assert len(rich_text[1]["text"]["content"]) == 770

    @pytest.mark.asyncio
    async def test_append_page_with_long_text_sends_multiple_children(self):
        """End-to-end regression: NotionAppendPageContentTool must chunk text > 2000 chars."""
        tool = NotionAppendPageContentTool()
        mock_response = {"results": [{"id": "b1"}, {"id": "b2"}]}
        long_text = "w" * 4500
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None,
                {"page_id": "p1", "text": long_text},
                credentials=make_credentials(),
            )

        called_body = mock_req.call_args[1]["json_body"]
        # Markdown converter wraps plain text in a single <p>, chunking happens inside rich_text
        assert len(called_body["children"]) == 1
        rich_text = called_body["children"][0]["paragraph"]["rich_text"]
        assert len(rich_text) == 3  # 2000 + 2000 + 500


# ─── Markdown to Notion Blocks ───────────────────────────────────────────────

@pytest.mark.unit
class TestMarkdownToNotionBlocks:
    """Test _markdown_to_notion_blocks converts markdown to structured Notion blocks."""

    def test_heading_1(self):
        blocks = _markdown_to_notion_blocks("# Hello World")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "heading_1"
        assert blocks[0]["heading_1"]["rich_text"][0]["text"]["content"] == "Hello World"

    def test_heading_2(self):
        blocks = _markdown_to_notion_blocks("## Sub Heading")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "heading_2"

    def test_heading_3(self):
        blocks = _markdown_to_notion_blocks("### Sub Sub Heading")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "heading_3"

    def test_h4_maps_to_heading_3(self):
        """Notion only has 3 heading levels; h4+ fall back to heading_3."""
        blocks = _markdown_to_notion_blocks("#### Deep Heading")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "heading_3"

    def test_bullet_list(self):
        md = "- first\n- second\n- third"
        blocks = _markdown_to_notion_blocks(md)
        assert len(blocks) == 3
        for b in blocks:
            assert b["type"] == "bulleted_list_item"
        assert blocks[0]["bulleted_list_item"]["rich_text"][0]["text"]["content"] == "first"
        assert blocks[2]["bulleted_list_item"]["rich_text"][0]["text"]["content"] == "third"

    def test_numbered_list(self):
        md = "1. alpha\n2. beta\n3. gamma"
        blocks = _markdown_to_notion_blocks(md)
        assert len(blocks) == 3
        for b in blocks:
            assert b["type"] == "numbered_list_item"

    def test_code_block(self):
        md = "```python\nprint('hello')\n```"
        blocks = _markdown_to_notion_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["code"]["language"] == "python"
        assert "print('hello')" in blocks[0]["code"]["rich_text"][0]["text"]["content"]

    def test_code_block_no_language(self):
        md = "```\nsome code\n```"
        blocks = _markdown_to_notion_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["code"]["language"] == "plain text"

    def test_blockquote(self):
        md = "> This is a quote"
        blocks = _markdown_to_notion_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "quote"
        assert "This is a quote" in blocks[0]["quote"]["rich_text"][0]["text"]["content"]

    def test_horizontal_rule(self):
        md = "above\n\n---\n\nbelow"
        blocks = _markdown_to_notion_blocks(md)
        types = [b["type"] for b in blocks]
        assert "divider" in types

    def test_bold_annotation(self):
        md = "Some **bold** text"
        blocks = _markdown_to_notion_blocks(md)
        assert blocks[0]["type"] == "paragraph"
        rich_text = blocks[0]["paragraph"]["rich_text"]
        bold_parts = [rt for rt in rich_text if rt.get("annotations", {}).get("bold")]
        assert len(bold_parts) >= 1
        assert bold_parts[0]["text"]["content"] == "bold"

    def test_italic_annotation(self):
        md = "Some *italic* text"
        blocks = _markdown_to_notion_blocks(md)
        rich_text = blocks[0]["paragraph"]["rich_text"]
        italic_parts = [rt for rt in rich_text if rt.get("annotations", {}).get("italic")]
        assert len(italic_parts) >= 1
        assert italic_parts[0]["text"]["content"] == "italic"

    def test_inline_code_annotation(self):
        md = "Use `variable` here"
        blocks = _markdown_to_notion_blocks(md)
        rich_text = blocks[0]["paragraph"]["rich_text"]
        code_parts = [rt for rt in rich_text if rt.get("annotations", {}).get("code")]
        assert len(code_parts) >= 1
        assert code_parts[0]["text"]["content"] == "variable"

    def test_link(self):
        md = "Click [here](https://example.com) now"
        blocks = _markdown_to_notion_blocks(md)
        rich_text = blocks[0]["paragraph"]["rich_text"]
        link_parts = [rt for rt in rich_text if rt["text"].get("link")]
        assert len(link_parts) >= 1
        assert link_parts[0]["text"]["link"]["url"] == "https://example.com"
        assert link_parts[0]["text"]["content"] == "here"

    def test_mixed_content(self):
        md = "# Title\n\nSome paragraph.\n\n- bullet one\n- bullet two\n\n## Another heading"
        blocks = _markdown_to_notion_blocks(md)
        types = [b["type"] for b in blocks]
        assert types[0] == "heading_1"
        assert "paragraph" in types
        assert "bulleted_list_item" in types
        assert "heading_2" in types

    def test_plain_text_fallback(self):
        """Plain text without markdown syntax should produce paragraph blocks."""
        blocks = _markdown_to_notion_blocks("Just plain text here")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "paragraph"
        assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "Just plain text here"

    def test_empty_string_fallback(self):
        blocks = _markdown_to_notion_blocks("")
        assert len(blocks) >= 1
        assert blocks[0]["type"] == "paragraph"

    def test_long_content_chunked_in_rich_text(self):
        """Long plain text is still chunked at 2000-char boundaries inside rich_text."""
        long_text = "a" * 2770
        blocks = _markdown_to_notion_blocks(long_text)
        # Should be 1 paragraph with chunked rich_text
        assert len(blocks) == 1
        rich_text = blocks[0]["paragraph"]["rich_text"]
        total = sum(len(rt["text"]["content"]) for rt in rich_text)
        assert total == 2770

    @pytest.mark.asyncio
    async def test_create_page_with_markdown_produces_heading(self):
        """End-to-end: notion_create_page with markdown heading produces heading_1 block."""
        tool = NotionCreatePageTool()
        mock_response = {"id": "id", "url": "https://notion.so/id"}
        with patch.object(tool, "_make_request", new=AsyncMock(return_value=mock_response)) as mock_req:
            await tool.execute(
                None,
                {
                    "parent_id": "parent",
                    "parent_type": "page",
                    "title": "Test",
                    "content": "# My Heading\n\nSome text.",
                },
                credentials=make_credentials(),
            )
        called_body = mock_req.call_args[1]["json_body"]
        children = called_body["children"]
        assert children[0]["type"] == "heading_1"
        assert children[1]["type"] == "paragraph"


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
