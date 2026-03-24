"""
Unit tests for workflow catalog operations.

Tests the catalog service functions with proper mocking of dependencies.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.workflows import models as api_models
from seer.api.workflows.services import catalog


# =============================================================================
# List Node Types Tests
# =============================================================================


@pytest.mark.unit
class TestListNodeTypes:
    """Tests for list_node_types function."""

    @pytest.mark.asyncio
    async def test_list_node_types_returns_response(self):
        """Test list_node_types returns NodeTypeResponse."""
        result = await catalog.list_node_types()

        assert isinstance(result, api_models.NodeTypeResponse)
        assert hasattr(result, "node_types")
        assert isinstance(result.node_types, list)

    @pytest.mark.asyncio
    async def test_list_node_types_includes_agent_type(self):
        """Test agent node type is included in response (supersedes removed llm type)."""
        result = await catalog.list_node_types()

        types = [nt.type for nt in result.node_types]
        assert "llm" not in types
        assert "agent" in types

    @pytest.mark.asyncio
    async def test_list_node_types_includes_if_else_type(self):
        """Test if_else node type is included in response."""
        result = await catalog.list_node_types()

        types = [nt.type for nt in result.node_types]
        assert "if_else" in types

    @pytest.mark.asyncio
    async def test_list_node_types_includes_for_loop_type(self):
        """Test for_loop node type is included in response."""
        result = await catalog.list_node_types()

        types = [nt.type for nt in result.node_types]
        assert "for_loop" in types

    @pytest.mark.asyncio
    async def test_list_node_types_includes_mcp_type(self):
        """Test MCP node type is included in response."""
        result = await catalog.list_node_types()

        types = [nt.type for nt in result.node_types]
        assert "mcp" in types

    @pytest.mark.asyncio
    async def test_list_node_types_includes_hitl_type(self):
        """Test HITL (Human-In-The-Loop) node type is included in response."""
        result = await catalog.list_node_types()

        types = [nt.type for nt in result.node_types]
        assert "hitl" in types

        hitl_descriptor = next(nt for nt in result.node_types if nt.type == "hitl")
        assert hitl_descriptor.title == "Human Input"

        field_names = [f.name for f in hitl_descriptor.fields]
        assert "id" in field_names
        assert "title" in field_names
        assert "description" in field_names
        assert "display" in field_names
        assert "inputs" in field_names
        assert "timeout_seconds" in field_names

    @pytest.mark.asyncio
    async def test_list_node_types_field_descriptors_have_required_fields(self):
        """Test node type descriptors have properly structured fields."""
        result = await catalog.list_node_types()

        for node_type in result.node_types:
            assert isinstance(node_type, api_models.NodeTypeDescriptor)
            assert node_type.type
            assert node_type.title
            assert isinstance(node_type.fields, list)

            for field in node_type.fields:
                assert isinstance(field, api_models.NodeFieldDescriptor)
                assert field.name
                assert field.kind


# =============================================================================
# List Triggers Tests
# =============================================================================


@pytest.mark.unit
class TestListTriggers:
    """Tests for list_triggers function."""

    @pytest.mark.asyncio
    async def test_list_triggers_returns_trigger_descriptors(self):
        """Test listing triggers returns TriggerCatalogResponse."""
        mock_definition = MagicMock()
        mock_definition.key = "webhook.generic"
        mock_definition.title = "Webhook"
        mock_definition.provider = "generic"
        mock_definition.mode = "push"
        mock_definition.description = "Generic webhook trigger"
        mock_definition.schemas.event = {"type": "object"}
        mock_definition.schemas.filter = None
        mock_definition.schemas.config = None
        mock_definition.meta.requires_connection = False

        mock_user = MagicMock()

        with patch.object(catalog.trigger_registry, "all", return_value=[mock_definition]):
            with patch("seer.database.models_oauth.OAuthConnection") as mock_oauth:
                mock_oauth.filter.return_value.all = AsyncMock(return_value=[])
                result = await catalog.list_triggers(mock_user)

        assert isinstance(result, api_models.TriggerCatalogResponse)
        assert len(result.triggers) == 1
        assert result.triggers[0].key == "webhook.generic"
        assert result.triggers[0].title == "Webhook"
        assert result.triggers[0].is_connected is True  # No connection required

    @pytest.mark.asyncio
    async def test_list_triggers_includes_all_fields(self):
        """Test trigger descriptors include all required fields."""
        mock_definition = MagicMock()
        mock_definition.key = "schedule.cron"
        mock_definition.title = "Scheduled"
        mock_definition.provider = "cron"
        mock_definition.mode = "poll"
        mock_definition.description = "Cron-based schedule trigger"
        mock_definition.schemas.event = {"type": "object", "properties": {"time": {"type": "string"}}}
        mock_definition.schemas.filter = {"type": "object"}
        mock_definition.schemas.config = {"type": "object", "properties": {"cron": {"type": "string"}}}
        mock_definition.meta.requires_connection = False

        mock_user = MagicMock()

        with patch.object(catalog.trigger_registry, "all", return_value=[mock_definition]):
            with patch("seer.database.models_oauth.OAuthConnection") as mock_oauth:
                mock_oauth.filter.return_value.all = AsyncMock(return_value=[])
                result = await catalog.list_triggers(mock_user)

        trigger = result.triggers[0]
        assert trigger.key == "schedule.cron"
        assert trigger.title == "Scheduled"
        assert trigger.provider == "cron"
        assert trigger.mode == "poll"
        assert trigger.description == "Cron-based schedule trigger"
        assert trigger.event_schema is not None
        assert trigger.filter_schema is not None
        assert trigger.config_schema is not None
        assert trigger.is_connected is True

    @pytest.mark.asyncio
    async def test_list_triggers_empty_registry(self):
        """Test handling empty trigger registry."""
        mock_user = MagicMock()

        with patch.object(catalog.trigger_registry, "all", return_value=[]):
            with patch("seer.database.models_oauth.OAuthConnection") as mock_oauth:
                mock_oauth.filter.return_value.all = AsyncMock(return_value=[])
                result = await catalog.list_triggers(mock_user)

        assert result.triggers == []

    @pytest.mark.asyncio
    async def test_list_triggers_is_connected_true_when_user_has_connection(self):
        """Test is_connected is True when user has active OAuth connection."""
        mock_definition = MagicMock()
        mock_definition.key = "poll.gmail.email_received"
        mock_definition.title = "Gmail"
        mock_definition.provider = "gmail"
        mock_definition.mode = "poll"
        mock_definition.description = "Gmail trigger"
        mock_definition.schemas.event = {"type": "object"}
        mock_definition.schemas.filter = None
        mock_definition.schemas.config = None
        mock_definition.meta.requires_connection = True
        mock_definition.meta.required_scopes = None  # No scope requirements

        mock_user = MagicMock()

        # Create mock connection with provider and scopes
        mock_connection = MagicMock()
        mock_connection.provider = "google"
        mock_connection.scopes = ""

        with patch.object(catalog.trigger_registry, "all", return_value=[mock_definition]):
            with patch("seer.database.models_oauth.OAuthConnection") as mock_oauth:
                # User has google connection (gmail maps to google)
                mock_oauth.filter.return_value.all = AsyncMock(return_value=[mock_connection])
                with patch("seer.services.integrations.auth.oauth.get_oauth_provider", return_value="google"):
                    result = await catalog.list_triggers(mock_user)

        assert result.triggers[0].is_connected is True

    @pytest.mark.asyncio
    async def test_list_triggers_is_connected_false_when_no_connection(self):
        """Test is_connected is False when user lacks OAuth connection."""
        mock_definition = MagicMock()
        mock_definition.key = "poll.discord.message_received"
        mock_definition.title = "Discord"
        mock_definition.provider = "discord"
        mock_definition.mode = "poll"
        mock_definition.description = "Discord trigger"
        mock_definition.schemas.event = {"type": "object"}
        mock_definition.schemas.filter = None
        mock_definition.schemas.config = None
        mock_definition.meta.requires_connection = True
        mock_definition.meta.required_scopes = None

        mock_user = MagicMock()

        with patch.object(catalog.trigger_registry, "all", return_value=[mock_definition]):
            with patch("seer.database.models_oauth.OAuthConnection") as mock_oauth:
                # User has no connections
                mock_oauth.filter.return_value.all = AsyncMock(return_value=[])
                with patch("seer.services.integrations.auth.oauth.get_oauth_provider", return_value="discord"):
                    result = await catalog.list_triggers(mock_user)

        assert result.triggers[0].is_connected is False

    @pytest.mark.asyncio
    async def test_list_triggers_is_connected_false_when_missing_required_scopes(self):
        """Test is_connected is False when user has connection but missing required scopes."""
        mock_definition = MagicMock()
        mock_definition.key = "poll.slack.message_received"
        mock_definition.title = "Slack"
        mock_definition.provider = "slack"
        mock_definition.mode = "poll"
        mock_definition.description = "Slack trigger"
        mock_definition.schemas.event = {"type": "object"}
        mock_definition.schemas.filter = None
        mock_definition.schemas.config = None
        mock_definition.meta.requires_connection = True
        mock_definition.meta.required_scopes = ["channels:history", "groups:history"]

        mock_user = MagicMock()

        # Create mock connection with wrong scopes
        mock_connection = MagicMock()
        mock_connection.provider = "slack"
        mock_connection.scopes = "chat:write channels:read"  # Missing required scopes

        with patch.object(catalog.trigger_registry, "all", return_value=[mock_definition]):
            with patch("seer.database.models_oauth.OAuthConnection") as mock_oauth:
                mock_oauth.filter.return_value.all = AsyncMock(return_value=[mock_connection])
                with patch("seer.services.integrations.auth.oauth.get_oauth_provider", return_value="slack"):
                    result = await catalog.list_triggers(mock_user)

        assert result.triggers[0].is_connected is False

    @pytest.mark.asyncio
    async def test_list_triggers_is_connected_true_when_has_required_scopes(self):
        """Test is_connected is True when user has connection with required scopes."""
        mock_definition = MagicMock()
        mock_definition.key = "poll.slack.message_received"
        mock_definition.title = "Slack"
        mock_definition.provider = "slack"
        mock_definition.mode = "poll"
        mock_definition.description = "Slack trigger"
        mock_definition.schemas.event = {"type": "object"}
        mock_definition.schemas.filter = None
        mock_definition.schemas.config = None
        mock_definition.meta.requires_connection = True
        mock_definition.meta.required_scopes = ["channels:history", "groups:history"]

        mock_user = MagicMock()

        # Create mock connection with correct scopes
        mock_connection = MagicMock()
        mock_connection.provider = "slack"
        mock_connection.scopes = "channels:history groups:history chat:write"

        with patch.object(catalog.trigger_registry, "all", return_value=[mock_definition]):
            with patch("seer.database.models_oauth.OAuthConnection") as mock_oauth:
                mock_oauth.filter.return_value.all = AsyncMock(return_value=[mock_connection])
                with patch("seer.services.integrations.auth.oauth.get_oauth_provider", return_value="slack"):
                    result = await catalog.list_triggers(mock_user)

        assert result.triggers[0].is_connected is True


# =============================================================================
# List Models Tests
# =============================================================================


@pytest.mark.unit
class TestListModels:
    """Tests for list_models function."""

    @pytest.mark.asyncio
    async def test_list_models_returns_default_models(self):
        """Test listing returns default model options."""
        with patch.object(catalog.shared_config, "default_llm_model", None):
            result = await catalog.list_models()

        assert isinstance(result, api_models.ModelRegistryResponse)
        assert len(result.models) >= 2

        model_ids = [m.id for m in result.models]
        assert "qwen/qwen3-235b-a22b-2507" in model_ids

    @pytest.mark.asyncio
    async def test_list_models_includes_custom_default(self):
        """Test custom default model is included when configured."""
        with patch.object(catalog.shared_config, "default_llm_model", "claude-3-opus"):
            result = await catalog.list_models()

        model_ids = [m.id for m in result.models]
        assert "claude-3-opus" in model_ids

    @pytest.mark.asyncio
    async def test_list_models_skips_duplicate_default(self):
        """Test duplicate default models are not added twice."""
        with patch.object(catalog.shared_config, "default_llm_model", "gpt-4.1-mini"):
            result = await catalog.list_models()

        # Count occurrences of gpt-4.1-mini
        count = sum(1 for m in result.models if m.id == "gpt-4.1-mini")
        assert count == 1

    @pytest.mark.asyncio
    async def test_list_models_descriptors_have_required_fields(self):
        """Test model descriptors have all required fields."""
        with patch.object(catalog.shared_config, "default_llm_model", None):
            result = await catalog.list_models()

        for model in result.models:
            assert isinstance(model, api_models.ModelDescriptor)
            assert model.id
            assert model.title
            assert isinstance(model.supports_json_schema, bool)


# =============================================================================
# List MCP Tools Tests
# =============================================================================


@pytest.mark.unit
class TestListMcpTools:
    """Tests for list_mcp_tools function."""

    @pytest.mark.asyncio
    async def test_list_mcp_tools_returns_tools(self):
        """Test listing MCP tools returns McpToolsResponse."""
        mock_tool = MagicMock()
        mock_tool.name = "mcp_tool_1"
        mock_tool.description = "A test MCP tool"
        mock_tool.inputSchema = {"type": "object"}

        payload = api_models.McpToolsRequest(
            server="http://localhost:8080",
            server_type="http"
        )

        with patch.object(
            catalog.COMPILER.mcp_client_registry,
            "list_tools",
            new_callable=AsyncMock,
            return_value=[mock_tool]
        ):
            result = await catalog.list_mcp_tools(payload)

        assert isinstance(result, api_models.McpToolsResponse)
        assert len(result.tools) == 1
        assert result.tools[0].name == "mcp_tool_1"
        assert result.tools[0].description == "A test MCP tool"

    @pytest.mark.asyncio
    async def test_list_mcp_tools_connection_error(self):
        """Test handling MCP connection errors."""
        payload = api_models.McpToolsRequest(
            server="http://invalid-server",
            server_type="http"
        )

        with patch.object(
            catalog.COMPILER.mcp_client_registry,
            "list_tools",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Connection refused")
        ):
            with pytest.raises(Exception):  # raise_problem raises exception
                await catalog.list_mcp_tools(payload)

    @pytest.mark.asyncio
    async def test_list_mcp_tools_with_auth(self):
        """Test listing MCP tools with authentication."""
        mock_tool = MagicMock()
        mock_tool.name = "auth_tool"
        mock_tool.description = "Authenticated tool"
        mock_tool.inputSchema = None

        payload = api_models.McpToolsRequest(
            server="http://localhost:8080",
            server_type="http",
            auth={"api_key": "secret"}
        )

        with patch.object(
            catalog.COMPILER.mcp_client_registry,
            "list_tools",
            new_callable=AsyncMock,
            return_value=[mock_tool]
        ):
            result = await catalog.list_mcp_tools(payload)

        assert len(result.tools) == 1
        assert result.tools[0].name == "auth_tool"
