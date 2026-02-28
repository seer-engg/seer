"""Unit tests for Airtable tools."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from seer.tools.airtable.bases import AirtableListBasesTool, AirtableListTablesTool
from seer.tools.airtable.records import (
    AirtableListRecordsTool,
    AirtableCreateRecordTool,
    AirtableUpdateRecordTool,
    AirtableDeleteRecordTool,
)


# =============================================================================
# AirtableListBasesTool Tests
# =============================================================================


@pytest.mark.unit
class TestAirtableListBasesTool:
    """Tests for AirtableListBasesTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = AirtableListBasesTool()
        assert tool.name == "airtable_list_bases"
        assert tool.integration_type == "airtable"
        assert tool.provider == "airtable"
        assert "schema.bases:read" in tool.required_scopes

    def test_get_parameters_schema(self):
        """Test parameter schema has expected structure."""
        tool = AirtableListBasesTool()
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "offset" in schema["properties"]
        assert schema["required"] == []

    def test_get_output_schema(self):
        """Test output schema has expected fields."""
        tool = AirtableListBasesTool()
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "bases" in schema["properties"]
        assert "offset" in schema["properties"]
        assert "bases" in schema["required"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful base listing."""
        tool = AirtableListBasesTool()

        mock_response = {
            "bases": [
                {"id": "appABC123", "name": "Project Tracker", "permissionLevel": "create"},
                {"id": "appDEF456", "name": "Content Calendar", "permissionLevel": "edit"},
            ]
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute("test_token", {})

            assert len(result["bases"]) == 2
            assert result["bases"][0]["id"] == "appABC123"
            assert result["bases"][0]["name"] == "Project Tracker"
            mock_request.assert_called_once()
            assert mock_request.call_args[0][1] == "meta/bases"


# =============================================================================
# AirtableListTablesTool Tests
# =============================================================================


@pytest.mark.unit
class TestAirtableListTablesTool:
    """Tests for AirtableListTablesTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = AirtableListTablesTool()
        assert tool.name == "airtable_list_tables"
        assert tool.integration_type == "airtable"
        assert "schema.bases:read" in tool.required_scopes

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = AirtableListTablesTool()
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "base_id" in schema["properties"]
        assert "base_id" in schema["required"]

    def test_get_resource_pickers(self):
        """Test resource pickers configuration."""
        tool = AirtableListTablesTool()
        pickers = tool.get_resource_pickers()

        assert "base_id" in pickers
        assert pickers["base_id"]["resource_type"] == "base"
        assert pickers["base_id"]["filter"]["provider"] == "airtable"

    @pytest.mark.asyncio
    async def test_execute_missing_base_id(self):
        """Test execute raises error when base_id is missing."""
        tool = AirtableListTablesTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {})

        assert exc_info.value.status_code == 400
        assert "base_id" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful table listing."""
        tool = AirtableListTablesTool()

        mock_response = {
            "tables": [
                {
                    "id": "tblTasks",
                    "name": "Tasks",
                    "primaryFieldId": "fldName",
                    "fields": [{"id": "fldName", "name": "Name", "type": "singleLineText"}],
                    "views": [{"id": "viwGrid", "name": "Grid view", "type": "grid"}],
                }
            ]
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute("test_token", {"base_id": "appABC123"})

            assert len(result["tables"]) == 1
            assert result["tables"][0]["name"] == "Tasks"
            mock_request.assert_called_once()


# =============================================================================
# AirtableListRecordsTool Tests
# =============================================================================


@pytest.mark.unit
class TestAirtableListRecordsTool:
    """Tests for AirtableListRecordsTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = AirtableListRecordsTool()
        assert tool.name == "airtable_list_records"
        assert tool.integration_type == "airtable"
        assert "data.records:read" in tool.required_scopes

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = AirtableListRecordsTool()
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "base_id" in schema["properties"]
        assert "table_id_or_name" in schema["properties"]
        assert "filter_by_formula" in schema["properties"]
        assert "max_records" in schema["properties"]
        assert "sort" in schema["properties"]
        assert "base_id" in schema["required"]
        assert "table_id_or_name" in schema["required"]

    def test_get_resource_pickers(self):
        """Test resource pickers configuration."""
        tool = AirtableListRecordsTool()
        pickers = tool.get_resource_pickers()

        assert "base_id" in pickers
        assert "table_id_or_name" in pickers
        assert pickers["table_id_or_name"]["depends_on"] == "base_id"

    @pytest.mark.asyncio
    async def test_execute_missing_base_id(self):
        """Test execute raises error when base_id is missing."""
        tool = AirtableListRecordsTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {"table_id_or_name": "Tasks"})

        assert exc_info.value.status_code == 400
        assert "base_id" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_missing_table_id(self):
        """Test execute raises error when table_id_or_name is missing."""
        tool = AirtableListRecordsTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {"base_id": "appABC123"})

        assert exc_info.value.status_code == 400
        assert "table_id_or_name" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful record listing."""
        tool = AirtableListRecordsTool()

        mock_response = {
            "records": [
                {
                    "id": "recABC123",
                    "createdTime": "2024-01-15T10:00:00.000Z",
                    "fields": {"Name": "Task 1", "Status": "Done"},
                },
                {
                    "id": "recDEF456",
                    "createdTime": "2024-01-16T10:00:00.000Z",
                    "fields": {"Name": "Task 2", "Status": "In Progress"},
                },
            ]
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                "test_token",
                {"base_id": "appABC123", "table_id_or_name": "Tasks"},
            )

            assert len(result["records"]) == 2
            # Records should be flattened
            assert result["records"][0]["id"] == "recABC123"
            assert result["records"][0]["Name"] == "Task 1"
            assert result["records"][0]["Status"] == "Done"


# =============================================================================
# AirtableCreateRecordTool Tests
# =============================================================================


@pytest.mark.unit
class TestAirtableCreateRecordTool:
    """Tests for AirtableCreateRecordTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = AirtableCreateRecordTool()
        assert tool.name == "airtable_create_record"
        assert tool.integration_type == "airtable"
        assert "data.records:write" in tool.required_scopes

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = AirtableCreateRecordTool()
        schema = tool.get_parameters_schema()

        assert "base_id" in schema["properties"]
        assert "table_id_or_name" in schema["properties"]
        assert "fields" in schema["properties"]
        assert "records" in schema["properties"]

    @pytest.mark.asyncio
    async def test_execute_missing_fields_and_records(self):
        """Test execute raises error when neither fields nor records provided."""
        tool = AirtableCreateRecordTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                "test_token",
                {"base_id": "appABC123", "table_id_or_name": "Tasks"},
            )

        assert exc_info.value.status_code == 400
        assert "fields" in exc_info.value.detail or "records" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_both_fields_and_records(self):
        """Test execute raises error when both fields and records provided."""
        tool = AirtableCreateRecordTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                "test_token",
                {
                    "base_id": "appABC123",
                    "table_id_or_name": "Tasks",
                    "fields": {"Name": "Task 1"},
                    "records": [{"fields": {"Name": "Task 2"}}],
                },
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_max_records_limit(self):
        """Test execute raises error when more than 10 records provided."""
        tool = AirtableCreateRecordTool()

        records = [{"fields": {"Name": f"Task {i}"}} for i in range(11)]

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                "test_token",
                {
                    "base_id": "appABC123",
                    "table_id_or_name": "Tasks",
                    "records": records,
                },
            )

        assert exc_info.value.status_code == 400
        assert "10" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_single_record_success(self):
        """Test successful single record creation."""
        tool = AirtableCreateRecordTool()

        mock_response = {
            "records": [
                {
                    "id": "recNEW123",
                    "createdTime": "2024-01-20T10:00:00.000Z",
                    "fields": {"Name": "New Task", "Status": "Todo"},
                }
            ]
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                "test_token",
                {
                    "base_id": "appABC123",
                    "table_id_or_name": "Tasks",
                    "fields": {"Name": "New Task", "Status": "Todo"},
                },
            )

            assert len(result["records"]) == 1
            assert result["records"][0]["id"] == "recNEW123"

            # Verify POST method was used
            assert mock_request.call_args[0][0] == "POST"


# =============================================================================
# AirtableUpdateRecordTool Tests
# =============================================================================


@pytest.mark.unit
class TestAirtableUpdateRecordTool:
    """Tests for AirtableUpdateRecordTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = AirtableUpdateRecordTool()
        assert tool.name == "airtable_update_record"
        assert tool.integration_type == "airtable"
        assert "data.records:write" in tool.required_scopes

    @pytest.mark.asyncio
    async def test_execute_missing_record_info(self):
        """Test execute raises error when no record info provided."""
        tool = AirtableUpdateRecordTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                "test_token",
                {"base_id": "appABC123", "table_id_or_name": "Tasks"},
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_single_record_success(self):
        """Test successful single record update."""
        tool = AirtableUpdateRecordTool()

        mock_response = {
            "records": [
                {
                    "id": "recABC123",
                    "createdTime": "2024-01-15T10:00:00.000Z",
                    "fields": {"Name": "Updated Task", "Status": "Done"},
                }
            ]
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                "test_token",
                {
                    "base_id": "appABC123",
                    "table_id_or_name": "Tasks",
                    "record_id": "recABC123",
                    "fields": {"Status": "Done"},
                },
            )

            assert len(result["records"]) == 1
            assert result["records"][0]["Status"] == "Done"

            # Verify PATCH method was used
            assert mock_request.call_args[0][0] == "PATCH"


# =============================================================================
# AirtableDeleteRecordTool Tests
# =============================================================================


@pytest.mark.unit
class TestAirtableDeleteRecordTool:
    """Tests for AirtableDeleteRecordTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = AirtableDeleteRecordTool()
        assert tool.name == "airtable_delete_record"
        assert tool.integration_type == "airtable"
        assert "data.records:write" in tool.required_scopes

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = AirtableDeleteRecordTool()
        schema = tool.get_parameters_schema()

        assert "base_id" in schema["properties"]
        assert "table_id_or_name" in schema["properties"]
        assert "record_id" in schema["properties"]
        assert "record_ids" in schema["properties"]

    @pytest.mark.asyncio
    async def test_execute_missing_record_ids(self):
        """Test execute raises error when no record IDs provided."""
        tool = AirtableDeleteRecordTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                "test_token",
                {"base_id": "appABC123", "table_id_or_name": "Tasks"},
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_both_record_id_and_record_ids(self):
        """Test execute raises error when both record_id and record_ids provided."""
        tool = AirtableDeleteRecordTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                "test_token",
                {
                    "base_id": "appABC123",
                    "table_id_or_name": "Tasks",
                    "record_id": "recABC123",
                    "record_ids": ["recDEF456"],
                },
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_max_records_limit(self):
        """Test execute raises error when more than 10 record IDs provided."""
        tool = AirtableDeleteRecordTool()

        record_ids = [f"rec{i}" for i in range(11)]

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                "test_token",
                {
                    "base_id": "appABC123",
                    "table_id_or_name": "Tasks",
                    "record_ids": record_ids,
                },
            )

        assert exc_info.value.status_code == 400
        assert "10" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_single_record_success(self):
        """Test successful single record deletion."""
        tool = AirtableDeleteRecordTool()

        mock_response = {
            "records": [
                {"id": "recABC123", "deleted": True}
            ]
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                "test_token",
                {
                    "base_id": "appABC123",
                    "table_id_or_name": "Tasks",
                    "record_id": "recABC123",
                },
            )

            assert len(result["records"]) == 1
            assert result["records"][0]["id"] == "recABC123"
            assert result["records"][0]["deleted"] is True

            # Verify DELETE method was used
            assert mock_request.call_args[0][0] == "DELETE"
