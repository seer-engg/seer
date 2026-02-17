"""
Unit tests for api.forms.router module.

Tests form resolution and submission endpoints.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# =============================================================================
# Resolve Form Tests
# =============================================================================


@pytest.mark.unit
class TestResolveForm:
    """Tests for GET /forms/resolve/{suffix} endpoint."""

    @pytest.mark.asyncio
    async def test_resolve_form_success(self):
        """Test resolving existing form by suffix."""
        from seer.api.forms.router import resolve_form

        mock_subscription = MagicMock()
        mock_subscription.id = 123
        mock_subscription.form_config = {
            "title": "Contact Form",
            "description": "Get in touch with us",
            "submitButtonText": "Send Message",
            "successMessage": "Thanks for reaching out!",
            "styling": {"theme": "light"},
        }
        mock_subscription.form_fields = [
            {"name": "email", "type": "email", "required": True},
            {"name": "message", "type": "textarea", "required": True},
        ]

        with patch("seer.api.forms.router.TriggerSubscription") as MockSubscription:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=mock_subscription)
            MockSubscription.filter = MagicMock(return_value=mock_query)

            result = await resolve_form("contact-form")

        assert result["form_id"] == 123
        assert result["title"] == "Contact Form"
        assert result["description"] == "Get in touch with us"
        assert result["submit_button_text"] == "Send Message"
        assert result["success_message"] == "Thanks for reaching out!"
        assert len(result["fields"]) == 2

    @pytest.mark.asyncio
    async def test_resolve_form_not_found(self):
        """Test resolving non-existent form raises 404."""
        from seer.api.forms.router import resolve_form

        with patch("seer.api.forms.router.TriggerSubscription") as MockSubscription:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=None)
            MockSubscription.filter = MagicMock(return_value=mock_query)

            with pytest.raises(HTTPException) as exc_info:
                await resolve_form("nonexistent-form")

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Form not found"

    @pytest.mark.asyncio
    async def test_resolve_form_default_values(self):
        """Test resolving form with minimal config uses defaults."""
        from seer.api.forms.router import resolve_form

        mock_subscription = MagicMock()
        mock_subscription.id = 456
        mock_subscription.form_config = {}  # Empty config
        mock_subscription.form_fields = []

        with patch("seer.api.forms.router.TriggerSubscription") as MockSubscription:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=mock_subscription)
            MockSubscription.filter = MagicMock(return_value=mock_query)

            result = await resolve_form("minimal-form")

        assert result["title"] == "Form"  # Default
        assert result["description"] is None
        assert result["submit_button_text"] == "Submit"  # Default
        assert result["success_message"] == "Thank you for your submission!"  # Default

    @pytest.mark.asyncio
    async def test_resolve_form_filters_by_enabled(self):
        """Test resolve_form only finds enabled subscriptions."""
        from seer.api.forms.router import resolve_form

        with patch("seer.api.forms.router.TriggerSubscription") as MockSubscription:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=None)
            MockSubscription.filter = MagicMock(return_value=mock_query)

            with pytest.raises(HTTPException):
                await resolve_form("test-form")

        # Verify filter was called with enabled=True
        MockSubscription.filter.assert_called_once()
        call_kwargs = MockSubscription.filter.call_args[1]
        assert call_kwargs["form_suffix"] == "test-form"
        assert call_kwargs["enabled"] is True


# =============================================================================
# Submit Form Tests
# =============================================================================


@pytest.mark.unit
class TestSubmitForm:
    """Tests for POST /forms/submit/{suffix} endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = MagicMock()
        request.json = AsyncMock(return_value={"email": "test@example.com", "message": "Hello"})
        request.headers = {"content-type": "application/json"}
        return request

    @pytest.mark.asyncio
    async def test_submit_form_success(self, mock_request):
        """Test successful form submission."""
        from seer.api.forms.router import submit_form

        mock_subscription = MagicMock()
        mock_subscription.id = 123
        mock_subscription.form_config = {}  # Regular form, not HITL
        mock_subscription.form_fields = [
            {"name": "email", "type": "email", "required": True},
            {"name": "message", "type": "text", "required": True},
        ]

        mock_event = MagicMock()
        mock_event.id = 456

        with patch("seer.api.forms.router.TriggerSubscription") as MockSubscription:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=mock_subscription)
            MockSubscription.filter = MagicMock(return_value=mock_query)

            with patch("seer.api.forms.router.validate_form_data", return_value=[]):
                with patch("seer.api.forms.router.handle_generic_webhook", new_callable=AsyncMock, return_value=mock_event):
                    result = await submit_form("contact-form", mock_request)

        assert result["ok"] is True
        assert result["event_id"] == 456
        assert result["message"] == "Form submitted successfully"

    @pytest.mark.asyncio
    async def test_submit_form_not_found(self, mock_request):
        """Test submitting to non-existent form raises 404."""
        from seer.api.forms.router import submit_form

        with patch("seer.api.forms.router.TriggerSubscription") as MockSubscription:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=None)
            MockSubscription.filter = MagicMock(return_value=mock_query)

            with pytest.raises(HTTPException) as exc_info:
                await submit_form("nonexistent-form", mock_request)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_submit_form_validation_errors(self, mock_request):
        """Test form submission with validation errors raises 400."""
        from seer.api.forms.router import submit_form

        mock_subscription = MagicMock()
        mock_subscription.id = 123
        mock_subscription.form_config = {}  # Regular form, not HITL
        mock_subscription.form_fields = [
            {"name": "email", "type": "email", "required": True},
        ]

        with patch("seer.api.forms.router.TriggerSubscription") as MockSubscription:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=mock_subscription)
            MockSubscription.filter = MagicMock(return_value=mock_query)

            with patch("seer.api.forms.router.validate_form_data", return_value=[{"field": "email", "error": "Invalid email"}]):
                with pytest.raises(HTTPException) as exc_info:
                    await submit_form("contact-form", mock_request)

        assert exc_info.value.status_code == 400
        assert "errors" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_submit_form_invalid_json(self):
        """Test form submission with invalid JSON raises 400."""
        from seer.api.forms.router import submit_form

        mock_request = MagicMock()
        mock_request.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

        mock_subscription = MagicMock()
        mock_subscription.id = 123
        mock_subscription.form_fields = []

        with patch("seer.api.forms.router.TriggerSubscription") as MockSubscription:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=mock_subscription)
            MockSubscription.filter = MagicMock(return_value=mock_query)

            with pytest.raises(HTTPException) as exc_info:
                await submit_form("contact-form", mock_request)

        assert exc_info.value.status_code == 400
        assert "Invalid JSON" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_submit_form_with_empty_fields_list(self, mock_request):
        """Test form submission with no field validation."""
        from seer.api.forms.router import submit_form

        mock_subscription = MagicMock()
        mock_subscription.id = 789
        mock_subscription.form_config = {}  # Regular form, not HITL
        mock_subscription.form_fields = None  # No fields defined

        mock_event = MagicMock()
        mock_event.id = 999

        with patch("seer.api.forms.router.TriggerSubscription") as MockSubscription:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=mock_subscription)
            MockSubscription.filter = MagicMock(return_value=mock_query)

            with patch("seer.api.forms.router.validate_form_data", return_value=[]):
                with patch("seer.api.forms.router.handle_generic_webhook", new_callable=AsyncMock, return_value=mock_event):
                    result = await submit_form("any-form", mock_request)

        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_submit_form_webhook_error(self, mock_request):
        """Test form submission handles webhook errors gracefully."""
        from seer.api.forms.router import submit_form

        mock_subscription = MagicMock()
        mock_subscription.id = 123
        mock_subscription.form_config = {}  # Regular form, not HITL
        mock_subscription.form_fields = []

        with patch("seer.api.forms.router.TriggerSubscription") as MockSubscription:
            mock_query = MagicMock()
            mock_query.first = AsyncMock(return_value=mock_subscription)
            MockSubscription.filter = MagicMock(return_value=mock_query)

            with patch("seer.api.forms.router.validate_form_data", return_value=[]):
                with patch("seer.api.forms.router.handle_generic_webhook", new_callable=AsyncMock, side_effect=RuntimeError("Webhook failed")):
                    with pytest.raises(HTTPException) as exc_info:
                        await submit_form("contact-form", mock_request)

        assert exc_info.value.status_code == 500
        assert "Failed to submit form" in exc_info.value.detail
