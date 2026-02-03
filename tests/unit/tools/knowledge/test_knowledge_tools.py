"""Unit tests for knowledge base tools."""

import pytest

from seer.tools.knowledge.query import KnowledgeBaseQueryTool
from seer.tools.knowledge.add import KnowledgeBaseAddTextTool
from seer.tools.knowledge.list import KnowledgeBaseListTool


class TestKnowledgeBaseQueryTool:
    """Test KnowledgeBaseQueryTool metadata and schema."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return KnowledgeBaseQueryTool()

    def test_tool_name(self, tool):
        """Test tool name is correct."""
        assert tool.name == "kb_query"

    def test_tool_description(self, tool):
        """Test tool has description."""
        assert tool.description
        assert "knowledge base" in tool.description.lower()
        assert "semantic search" in tool.description.lower()

    def test_tool_integration_type(self, tool):
        """Test tool integration type."""
        assert tool.integration_type == "knowledge"

    def test_parameters_schema(self, tool):
        """Test parameter schema structure."""
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Required parameters
        assert "kb_id" in schema["required"]
        assert "query" in schema["required"]

        # Parameter definitions
        props = schema["properties"]
        assert "kb_id" in props
        assert "query" in props
        assert "top_k" in props
        assert "min_score" in props

        # Type checks
        assert props["kb_id"]["type"] == "string"
        assert props["query"]["type"] == "string"
        assert props["top_k"]["type"] == "integer"
        assert props["min_score"]["type"] == "number"

    def test_output_schema(self, tool):
        """Test output schema structure."""
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "results" in schema["properties"]


class TestKnowledgeBaseAddTextTool:
    """Test KnowledgeBaseAddTextTool metadata and schema."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return KnowledgeBaseAddTextTool()

    def test_tool_name(self, tool):
        """Test tool name is correct."""
        assert tool.name == "kb_add_text"

    def test_tool_description(self, tool):
        """Test tool has description."""
        assert tool.description
        assert "knowledge base" in tool.description.lower()

    def test_tool_integration_type(self, tool):
        """Test tool integration type."""
        assert tool.integration_type == "knowledge"

    def test_parameters_schema(self, tool):
        """Test parameter schema structure."""
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Required parameters
        assert "kb_id" in schema["required"]
        assert "content" in schema["required"]
        assert "name" in schema["required"]

        # Parameter definitions
        props = schema["properties"]
        assert "kb_id" in props
        assert "content" in props
        assert "name" in props
        assert "metadata" in props

    def test_output_schema(self, tool):
        """Test output schema structure."""
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "success" in schema["properties"]
        assert "doc_id" in schema["properties"]
        assert "chunk_count" in schema["properties"]


class TestKnowledgeBaseListTool:
    """Test KnowledgeBaseListTool metadata and schema."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return KnowledgeBaseListTool()

    def test_tool_name(self, tool):
        """Test tool name is correct."""
        assert tool.name == "kb_list"

    def test_tool_description(self, tool):
        """Test tool has description."""
        assert tool.description
        assert "knowledge base" in tool.description.lower()

    def test_tool_integration_type(self, tool):
        """Test tool integration type."""
        assert tool.integration_type == "knowledge"

    def test_parameters_schema(self, tool):
        """Test parameter schema has no required parameters."""
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert schema["required"] == []

    def test_output_schema(self, tool):
        """Test output schema structure."""
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "knowledge_bases" in schema["properties"]
        assert "total" in schema["properties"]


class TestToolRegistration:
    """Test that all knowledge tools can be registered."""

    def test_tools_can_be_imported(self):
        """Test that tools module can be imported."""
        from seer.tools.knowledge import (
            KnowledgeBaseQueryTool,
            KnowledgeBaseAddTextTool,
            KnowledgeBaseListTool,
            register_knowledge_tools,
        )

        assert KnowledgeBaseQueryTool is not None
        assert KnowledgeBaseAddTextTool is not None
        assert KnowledgeBaseListTool is not None
        assert callable(register_knowledge_tools)

    def test_tools_have_metadata(self):
        """Test that all tools have proper metadata."""
        tools = [
            KnowledgeBaseQueryTool(),
            KnowledgeBaseAddTextTool(),
            KnowledgeBaseListTool(),
        ]

        for tool in tools:
            metadata = tool.get_metadata()
            assert "name" in metadata
            assert "description" in metadata
            assert "parameters" in metadata
            assert metadata["integration_type"] == "knowledge"
