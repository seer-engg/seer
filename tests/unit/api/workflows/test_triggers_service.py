"""
Unit tests for api.workflows.services.triggers module.

Tests trigger subscription management, validation, and webhook URL building.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Auto Select Provider Connection Tests
# =============================================================================


@pytest.mark.unit
class TestAutoSelectProviderConnection:
    """Tests for _auto_select_provider_connection function."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock user."""
        user = MagicMock()
        user.id = 1
        user.user_id = "test_user_123"
        return user

    @pytest.fixture
    def mock_trigger_definition(self):
        """Create a mock trigger definition."""
        definition = MagicMock()
        definition.key = "gmail.new_email"
        definition.provider = "gmail"
        return definition

    @pytest.mark.asyncio
    async def test_auto_select_finds_connection(self, mock_user, mock_trigger_definition):
        """Test auto-select finds most recent active connection."""
        from seer.api.workflows.services.triggers import _auto_select_provider_connection

        mock_connection = MagicMock()
        mock_connection.id = 123

        with patch("seer.api.workflows.services.triggers.get_oauth_provider", return_value="google"):
            with patch("seer.api.workflows.services.triggers.OAuthConnection") as MockOAuthConnection:
                mock_query = MagicMock()
                mock_query.order_by = MagicMock(return_value=MagicMock(first=AsyncMock(return_value=mock_connection)))
                MockOAuthConnection.filter = MagicMock(return_value=mock_query)

                result = await _auto_select_provider_connection(mock_user, mock_trigger_definition)

        assert result == 123

    @pytest.mark.asyncio
    async def test_auto_select_returns_none_when_no_connection(self, mock_user, mock_trigger_definition):
        """Test auto-select returns None when no active connection found."""
        from seer.api.workflows.services.triggers import _auto_select_provider_connection

        with patch("seer.api.workflows.services.triggers.get_oauth_provider", return_value="google"):
            with patch("seer.api.workflows.services.triggers.OAuthConnection") as MockOAuthConnection:
                mock_query = MagicMock()
                mock_query.order_by = MagicMock(return_value=MagicMock(first=AsyncMock(return_value=None)))
                MockOAuthConnection.filter = MagicMock(return_value=mock_query)

                result = await _auto_select_provider_connection(mock_user, mock_trigger_definition)

        assert result is None


# =============================================================================
# Load Trigger Definition Tests
# =============================================================================


@pytest.mark.unit
class TestLoadTriggerDefinition:
    """Tests for _load_trigger_definition function."""

    def test_load_trigger_definition_success(self):
        """Test loading existing trigger definition."""
        from seer.api.workflows.services.triggers import _load_trigger_definition

        mock_definition = MagicMock()
        mock_definition.key = "webhook.generic"

        with patch("seer.api.workflows.services.triggers.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = mock_definition

            result = _load_trigger_definition("webhook.generic")

        assert result == mock_definition
        mock_registry.maybe_get.assert_called_once_with("webhook.generic")

    def test_load_trigger_definition_not_found_raises_problem(self):
        """Test loading non-existent trigger raises problem."""
        from seer.api.workflows.services.triggers import _load_trigger_definition

        with patch("seer.api.workflows.services.triggers.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = None
            with patch("seer.api.workflows.services.triggers._raise_problem") as mock_raise:
                mock_raise.side_effect = Exception("Problem raised")

                with pytest.raises(Exception, match="Problem raised"):
                    _load_trigger_definition("nonexistent.trigger")

        mock_raise.assert_called_once()
        call_kwargs = mock_raise.call_args[1]
        assert call_kwargs["status"] == 404


# =============================================================================
# Validate Filters Payload Tests
# =============================================================================


@pytest.mark.unit
class TestValidateFiltersPayload:
    """Tests for _validate_filters_payload function."""

    def test_validate_filters_empty_filters_passes(self):
        """Test empty filters always pass validation."""
        from seer.api.workflows.services.triggers import _validate_filters_payload

        mock_definition = MagicMock()
        mock_definition.schemas.filter = {"type": "object"}

        # Should not raise
        _validate_filters_payload({}, mock_definition)
        _validate_filters_payload(None, mock_definition)

    def test_validate_filters_no_schema_passes(self):
        """Test validation passes when no filter schema defined."""
        from seer.api.workflows.services.triggers import _validate_filters_payload

        mock_definition = MagicMock()
        mock_definition.schemas.filter = None

        # Should not raise
        _validate_filters_payload({"any": "value"}, mock_definition)

    def test_validate_filters_valid_filters_pass(self):
        """Test valid filters pass schema validation."""
        from seer.api.workflows.services.triggers import _validate_filters_payload

        mock_definition = MagicMock()
        mock_definition.schemas.filter = {
            "type": "object",
            "properties": {
                "status": {"type": "string"}
            }
        }

        # Should not raise
        _validate_filters_payload({"status": "active"}, mock_definition)

    def test_validate_filters_invalid_raises_problem(self):
        """Test invalid filters raise validation problem."""
        from seer.api.workflows.services.triggers import _validate_filters_payload

        mock_definition = MagicMock()
        mock_definition.schemas.filter = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            }
        }

        with patch("seer.api.workflows.services.triggers._raise_problem") as mock_raise:
            mock_raise.side_effect = Exception("Validation failed")

            with pytest.raises(Exception, match="Validation failed"):
                _validate_filters_payload({"count": "not_an_integer"}, mock_definition)

        mock_raise.assert_called_once()
        call_kwargs = mock_raise.call_args[1]
        assert call_kwargs["status"] == 400


# =============================================================================
# Is Expression Tests
# =============================================================================


@pytest.mark.unit
class TestIsExpression:
    """Tests for _is_expression function."""

    def test_is_expression_valid_expression(self):
        """Test valid expression pattern detection."""
        from seer.api.workflows.services.triggers import _is_expression

        assert _is_expression("${event.data.message}") is True
        assert _is_expression("  ${event.data.value}  ") is True  # With whitespace

    def test_is_expression_invalid_patterns(self):
        """Test non-expression values."""
        from seer.api.workflows.services.triggers import _is_expression

        assert _is_expression("plain_string") is False
        assert _is_expression("${incomplete") is False
        assert _is_expression("incomplete}") is False
        assert _is_expression(123) is False
        assert _is_expression(None) is False

    def test_is_expression_nested_braces(self):
        """Test expression with nested content."""
        from seer.api.workflows.services.triggers import _is_expression

        assert _is_expression("${event.data.nested.path}") is True


# =============================================================================
# Build Webhook URL Tests
# =============================================================================


@pytest.mark.unit
class TestBuildWebhookUrl:
    """Tests for _build_webhook_url function."""

    def test_build_webhook_url_generic(self):
        """Test building URL for generic webhook."""
        from seer.api.workflows.services.triggers import _build_webhook_url

        result = _build_webhook_url(123, "webhook.generic")
        assert result == "/v1/webhooks/generic/123"

    def test_build_webhook_url_supabase(self):
        """Test building URL for Supabase webhook."""
        from seer.api.workflows.services.triggers import _build_webhook_url

        result = _build_webhook_url(456, "webhook.supabase.db_changes")
        assert result == "/v1/webhooks/generic/456"

    def test_build_webhook_url_unknown_type(self):
        """Test building URL for unknown trigger type returns None."""
        from seer.api.workflows.services.triggers import _build_webhook_url

        result = _build_webhook_url(789, "schedule.cron")
        assert result is None


# =============================================================================
# Serialize Subscription Tests
# =============================================================================


@pytest.mark.unit
class TestSerializeSubscription:
    """Tests for _serialize_subscription function."""

    def test_serialize_subscription_basic_fields(self):
        """Test serialization includes all basic fields."""
        from seer.api.workflows.services.triggers import _serialize_subscription

        mock_subscription = MagicMock()
        mock_subscription.id = 1
        mock_subscription.workflow_id = 100
        mock_subscription.trigger_key = "schedule.cron"
        mock_subscription.provider_connection_id = 50
        mock_subscription.enabled = True
        mock_subscription.filters = {"key": "value"}
        mock_subscription.provider_config = {"cron": "* * * * *"}
        mock_subscription.secret_token = None
        mock_subscription.form_suffix = None
        mock_subscription.form_fields = None
        mock_subscription.form_config = None
        mock_subscription.created_at = MagicMock()
        mock_subscription.updated_at = MagicMock()

        with patch("seer.api.workflows.services.triggers.make_workflow_public_id", return_value="wf_abc123"):
            result = _serialize_subscription(mock_subscription)

        assert result.subscription_id == 1
        assert result.workflow_id == "wf_abc123"
        assert result.trigger_key == "schedule.cron"
        assert result.provider_connection_id == 50
        assert result.enabled is True
        assert result.filters == {"key": "value"}

    def test_serialize_subscription_with_webhook_url(self):
        """Test serialization includes webhook URL for webhook triggers."""
        from seer.api.workflows.services.triggers import _serialize_subscription

        mock_subscription = MagicMock()
        mock_subscription.id = 123
        mock_subscription.workflow_id = 100
        mock_subscription.trigger_key = "webhook.generic"
        mock_subscription.provider_connection_id = None
        mock_subscription.enabled = True
        mock_subscription.filters = {}
        mock_subscription.provider_config = {}
        mock_subscription.secret_token = "secret_abc"
        mock_subscription.form_suffix = None
        mock_subscription.form_fields = None
        mock_subscription.form_config = None
        mock_subscription.created_at = MagicMock()
        mock_subscription.updated_at = MagicMock()

        with patch("seer.api.workflows.services.triggers.make_workflow_public_id", return_value="wf_xyz"):
            result = _serialize_subscription(mock_subscription)

        assert result.webhook_url == "/v1/webhooks/generic/123"
        assert result.secret_token == "secret_abc"

    def test_serialize_subscription_with_form_url(self):
        """Test serialization includes form URL for form triggers."""
        from seer.api.workflows.services.triggers import _serialize_subscription

        mock_subscription = MagicMock()
        mock_subscription.id = 456
        mock_subscription.workflow_id = 200
        mock_subscription.trigger_key = "form.hosted"
        mock_subscription.provider_connection_id = None
        mock_subscription.enabled = True
        mock_subscription.filters = {}
        mock_subscription.provider_config = {}
        mock_subscription.secret_token = None
        mock_subscription.form_suffix = "contact-form"
        mock_subscription.form_fields = [{"name": "email", "type": "email"}]
        mock_subscription.form_config = {"title": "Contact Us"}
        mock_subscription.created_at = MagicMock()
        mock_subscription.updated_at = MagicMock()

        with patch("seer.api.workflows.services.triggers.make_workflow_public_id", return_value="wf_form"):
            with patch("seer.api.workflows.services.triggers.shared_config") as mock_config:
                mock_config.frontend_url = "https://app.example.com"
                result = _serialize_subscription(mock_subscription)

        assert result.form_url == "https://app.example.com/forms/contact-form"
        assert result.form_suffix == "contact-form"
        assert result.form_fields == [{"name": "email", "type": "email"}]


# =============================================================================
# Validate Form Suffix Tests
# =============================================================================


@pytest.mark.unit
class TestValidateFormSuffix:
    """Tests for _validate_form_suffix function."""

    def test_validate_form_suffix_valid(self):
        """Test valid form suffix passes validation."""
        from seer.api.workflows.services.triggers import _validate_form_suffix

        # Should not raise
        _validate_form_suffix("contact-form")
        _validate_form_suffix("my-form-123")
        _validate_form_suffix("form")
        _validate_form_suffix(None)

    def test_validate_form_suffix_invalid_chars_raises(self):
        """Test form suffix with invalid characters raises problem."""
        from seer.api.workflows.services.triggers import _validate_form_suffix

        with patch("seer.api.workflows.services.triggers._raise_problem") as mock_raise:
            mock_raise.side_effect = Exception("Invalid suffix")

            with pytest.raises(Exception, match="Invalid suffix"):
                _validate_form_suffix("UPPERCASE")

    def test_validate_form_suffix_reserved_raises(self):
        """Test reserved form suffix raises problem."""
        from seer.api.workflows.services.triggers import _validate_form_suffix

        with patch("seer.api.workflows.services.triggers._raise_problem") as mock_raise:
            mock_raise.side_effect = Exception("Reserved suffix")

            with pytest.raises(Exception, match="Reserved suffix"):
                _validate_form_suffix("admin")


# =============================================================================
# Extract Event Path Tests
# =============================================================================


@pytest.mark.unit
class TestExtractEventPath:
    """Tests for _extract_event_path function."""

    def test_extract_event_path_simple(self):
        """Test extracting simple event path."""
        from seer.api.workflows.services.triggers import _extract_event_path

        result = _extract_event_path("${event.message}")
        assert result == ["message"]

    def test_extract_event_path_nested(self):
        """Test extracting nested event path."""
        from seer.api.workflows.services.triggers import _extract_event_path

        result = _extract_event_path("${event.data.user.name}")
        assert result == ["data", "user", "name"]

    def test_extract_event_path_non_event_raises(self):
        """Test non-event reference raises ValueError."""
        from seer.api.workflows.services.triggers import _extract_event_path

        with pytest.raises(ValueError, match="must reference event"):
            _extract_event_path("${nodes.task1.output}")

    def test_extract_event_path_too_short_raises(self):
        """Test path with no property raises ValueError."""
        from seer.api.workflows.services.triggers import _extract_event_path

        # "${event}" doesn't start with "event." so raises "must reference event.*"
        with pytest.raises(ValueError, match="must reference event"):
            _extract_event_path("${event}")


# =============================================================================
# Evaluate Bindings Tests
# =============================================================================


@pytest.mark.unit
class TestEvaluateBindings:
    """Tests for _evaluate_bindings function."""

    def test_evaluate_bindings_expression(self):
        """Test evaluating expression bindings."""
        from seer.api.workflows.services.triggers import _evaluate_bindings

        bindings = {"user_email": "${event.data.email}"}
        event_payload = {"data": {"email": "test@example.com"}}

        result = _evaluate_bindings(bindings, event_payload)
        assert result == {"user_email": "test@example.com"}

    def test_evaluate_bindings_literal(self):
        """Test evaluating literal bindings."""
        from seer.api.workflows.services.triggers import _evaluate_bindings

        bindings = {"static_value": "hello"}
        event_payload = {"data": {}}

        result = _evaluate_bindings(bindings, event_payload)
        assert result == {"static_value": "hello"}

    def test_evaluate_bindings_mixed(self):
        """Test evaluating mixed expression and literal bindings."""
        from seer.api.workflows.services.triggers import _evaluate_bindings

        bindings = {
            "email": "${event.data.email}",
            "source": "webhook"
        }
        event_payload = {"data": {"email": "user@example.com"}}

        result = _evaluate_bindings(bindings, event_payload)
        assert result == {
            "email": "user@example.com",
            "source": "webhook"
        }

    def test_evaluate_bindings_empty(self):
        """Test evaluating empty/None bindings."""
        from seer.api.workflows.services.triggers import _evaluate_bindings

        assert _evaluate_bindings({}, {"data": {}}) == {}
        assert _evaluate_bindings(None, {"data": {}}) == {}


# =============================================================================
# Sync Trigger Subscriptions Tests
# =============================================================================


@pytest.mark.unit
class TestSyncTriggerSubscriptions:
    """Tests for sync_trigger_subscriptions function."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock user."""
        user = MagicMock()
        user.id = 1
        user.user_id = "test_user_123"
        return user

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow."""
        workflow = MagicMock()
        workflow.id = 100
        return workflow

    @pytest.fixture
    def mock_spec(self):
        """Create a mock workflow spec with a trigger."""
        spec = MagicMock()
        trigger = MagicMock()
        trigger.id = "trigger_1"
        trigger.key = "webhook.generic"
        trigger.filters = {}
        trigger.provider_config = {}
        trigger.ui_meta = None
        spec.triggers = [trigger]
        return spec

    @pytest.mark.asyncio
    async def test_sync_reenables_disabled_subscription(self, mock_user, mock_workflow, mock_spec):
        """
        Test that sync_trigger_subscriptions re-enables a previously disabled subscription.

        This is important for the "no published version" -> publish flow:
        1. Trigger fails with "no published version" and gets disabled
        2. User publishes workflow
        3. sync_trigger_subscriptions should re-enable the trigger
        """
        from seer.api.workflows.services.triggers import sync_trigger_subscriptions

        # Create a mock disabled subscription
        mock_subscription = MagicMock()
        mock_subscription.id = 1
        mock_subscription.trigger_id = "trigger_1"
        mock_subscription.trigger_key = "webhook.generic"
        mock_subscription.enabled = False  # Previously disabled due to no published version
        mock_subscription.secret_token = "existing_secret"
        mock_subscription.save = AsyncMock()

        # Mock trigger definition
        mock_definition = MagicMock()
        mock_definition.title = "Generic Webhook"
        mock_definition.meta.requires_connection = False
        mock_definition.schemas.filter = None

        with patch("seer.api.workflows.services.triggers.TriggerSubscription") as MockSub:
            # Return the disabled subscription as existing
            MockSub.filter = MagicMock(return_value=AsyncMock(return_value=[mock_subscription])())

            with patch("seer.api.workflows.services.triggers._load_trigger_definition", return_value=mock_definition):
                with patch("seer.api.workflows.services.triggers._validate_and_adjust_poll_interval", new_callable=AsyncMock, return_value=(60, None)):
                    with patch("seer.api.workflows.services.triggers.delete_trigger_subscription", new_callable=AsyncMock):
                        await sync_trigger_subscriptions(
                            mock_user, mock_workflow, mock_spec, skip_validation=True
                        )

        # Verify subscription was re-enabled
        assert mock_subscription.enabled is True
        mock_subscription.save.assert_called_once()
