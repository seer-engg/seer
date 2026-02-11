"""
Unit tests for HITL Email Service.

Tests the HITLEmailService which sends Gmail notifications with form links.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
class TestHITLEmailRendering:
    """Tests for email content rendering."""

    def test_render_email_html_includes_title(self):
        """Test that HTML email includes the HITL title."""
        from seer.services.workflows.hitl_email import HITLEmailService

        service = HITLEmailService()
        data = {
            "title": "Order Approval",
            "description": None,
            "display": [],
        }

        html = service._render_email_html(data, "https://example.com/forms/hitl-123")

        assert "Order Approval" in html
        assert "Action Required" not in html  # Should use actual title

    def test_render_email_html_includes_description(self):
        """Test that HTML email includes description when provided."""
        from seer.services.workflows.hitl_email import HITLEmailService

        service = HITLEmailService()
        data = {
            "title": "Review",
            "description": "Please review the attached document carefully.",
            "display": [],
        }

        html = service._render_email_html(data, "https://example.com/forms/hitl-123")

        assert "Please review the attached document carefully." in html

    def test_render_email_html_includes_display_items(self):
        """Test that HTML email includes display items."""
        from seer.services.workflows.hitl_email import HITLEmailService

        service = HITLEmailService()
        data = {
            "title": "Approval",
            "description": None,
            "display": [
                {"label": "Order ID", "value": "ORD-12345"},
                {"label": "Amount", "value": "$99.99"},
            ],
        }

        html = service._render_email_html(data, "https://example.com/forms/hitl-123")

        assert "Order ID" in html
        assert "ORD-12345" in html
        assert "Amount" in html
        assert "$99.99" in html

    def test_render_email_html_includes_form_link_button(self):
        """Test that HTML email includes CTA button with form link."""
        from seer.services.workflows.hitl_email import HITLEmailService

        service = HITLEmailService()
        data = {
            "title": "Action Required",
            "description": None,
            "display": [],
        }
        form_url = "https://app.example.com/forms/hitl-run_123-node_1"

        html = service._render_email_html(data, form_url)

        assert f'href="{form_url}"' in html
        assert "Respond Now" in html

    def test_render_email_html_escapes_html_chars(self):
        """Test that HTML special characters are escaped to prevent XSS."""
        from seer.services.workflows.hitl_email import HITLEmailService

        service = HITLEmailService()
        data = {
            "title": "Test <script>alert('xss')</script>",
            "description": None,
            "display": [
                {"label": "Input", "value": "<b>bold</b>"},
            ],
        }

        html = service._render_email_html(data, "https://example.com/forms/hitl-123")

        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;b&gt;bold&lt;/b&gt;" in html

    def test_render_email_text_includes_all_content(self):
        """Test that plain text email includes all content."""
        from seer.services.workflows.hitl_email import HITLEmailService

        service = HITLEmailService()
        data = {
            "title": "Approval Request",
            "description": "Please approve or reject this request.",
            "display": [
                {"label": "Request ID", "value": "REQ-001"},
            ],
        }
        form_url = "https://example.com/forms/hitl-123"

        text = service._render_email_text(data, form_url)

        assert "Approval Request" in text
        assert "Please approve or reject this request." in text
        assert "Request ID: REQ-001" in text
        assert form_url in text


@pytest.mark.unit
class TestHITLEmailSending:
    """Tests for sending HITL emails."""

    @pytest.mark.asyncio
    async def test_send_hitl_email_creates_form_first(self):
        """Test that sending email creates the HITL form first."""
        from seer.services.workflows.hitl_email import HITLEmailService
        from seer.core.schema.models import GmailDeliveryConfig

        with patch("seer.services.workflows.hitl_email.HITLFormService") as mock_form_cls, \
             patch("seer.services.workflows.hitl_email.get_oauth_token") as mock_get_token, \
             patch("seer.services.workflows.hitl_email.GmailSendEmailTool") as mock_gmail_cls:

            # Setup mocks
            mock_form_service = MagicMock()
            mock_form_service.create_hitl_form = AsyncMock(
                return_value=(MagicMock(id=1), "https://example.com/forms/hitl-123")
            )
            mock_form_cls.return_value = mock_form_service

            mock_get_token.return_value = (MagicMock(id=1), "access_token_123")

            mock_gmail_tool = MagicMock()
            mock_gmail_tool.execute = AsyncMock(return_value={"id": "msg_123"})
            mock_gmail_cls.return_value = mock_gmail_tool

            service = HITLEmailService()

            mock_user = MagicMock()
            mock_run = MagicMock()
            mock_run.run_id = "run_123"
            interrupt_data = {
                "type": "hitl",
                "node_id": "node_1",
                "title": "Test",
                "inputs": [],
            }
            gmail_config = GmailDeliveryConfig(recipient_email="user@example.com")

            await service.send_hitl_email(mock_user, mock_run, interrupt_data, gmail_config)

            # Verify form was created
            mock_form_service.create_hitl_form.assert_called_once_with(mock_run, interrupt_data)

    @pytest.mark.asyncio
    async def test_send_hitl_email_uses_gmail_tool(self):
        """Test that email is sent via Gmail tool."""
        from seer.services.workflows.hitl_email import HITLEmailService
        from seer.core.schema.models import GmailDeliveryConfig

        with patch("seer.services.workflows.hitl_email.HITLFormService") as mock_form_cls, \
             patch("seer.services.workflows.hitl_email.get_oauth_token") as mock_get_token, \
             patch("seer.services.workflows.hitl_email.GmailSendEmailTool") as mock_gmail_cls:

            mock_form_service = MagicMock()
            mock_form_service.create_hitl_form = AsyncMock(
                return_value=(MagicMock(id=1), "https://example.com/forms/hitl-123")
            )
            mock_form_cls.return_value = mock_form_service

            mock_get_token.return_value = (MagicMock(id=1), "access_token_456")

            mock_gmail_tool = MagicMock()
            mock_gmail_tool.execute = AsyncMock(return_value={"id": "msg_123"})
            mock_gmail_cls.return_value = mock_gmail_tool

            service = HITLEmailService()

            mock_user = MagicMock()
            mock_run = MagicMock()
            mock_run.run_id = "run_456"
            interrupt_data = {
                "type": "hitl",
                "node_id": "approval",
                "title": "Approval Required",
                "inputs": [],
            }
            gmail_config = GmailDeliveryConfig(recipient_email="approver@example.com")

            await service.send_hitl_email(mock_user, mock_run, interrupt_data, gmail_config)

            # Verify Gmail tool was called
            mock_gmail_tool.execute.assert_called_once()
            call_kwargs = mock_gmail_tool.execute.call_args.kwargs
            assert call_kwargs["access_token"] == "access_token_456"
            assert call_kwargs["arguments"]["to"] == ["approver@example.com"]
            assert "[Action Required]" in call_kwargs["arguments"]["subject"]

    @pytest.mark.asyncio
    async def test_send_hitl_email_uses_google_provider(self):
        """Test that OAuth token is fetched for Google provider."""
        from seer.services.workflows.hitl_email import HITLEmailService
        from seer.core.schema.models import GmailDeliveryConfig

        with patch("seer.services.workflows.hitl_email.HITLFormService") as mock_form_cls, \
             patch("seer.services.workflows.hitl_email.get_oauth_token") as mock_get_token, \
             patch("seer.services.workflows.hitl_email.GmailSendEmailTool") as mock_gmail_cls:

            mock_form_service = MagicMock()
            mock_form_service.create_hitl_form = AsyncMock(
                return_value=(MagicMock(id=1), "https://example.com/forms/hitl-123")
            )
            mock_form_cls.return_value = mock_form_service

            mock_get_token.return_value = (MagicMock(id=1), "token")

            mock_gmail_tool = MagicMock()
            mock_gmail_tool.execute = AsyncMock(return_value={"id": "msg"})
            mock_gmail_cls.return_value = mock_gmail_tool

            service = HITLEmailService()

            mock_user = MagicMock()
            mock_run = MagicMock()
            mock_run.run_id = "run_789"
            interrupt_data = {"type": "hitl", "node_id": "n", "title": "T", "inputs": []}
            gmail_config = GmailDeliveryConfig(recipient_email="test@example.com")

            await service.send_hitl_email(mock_user, mock_run, interrupt_data, gmail_config)

            # Verify Google provider was used
            mock_get_token.assert_called_once_with(mock_user, provider="google")


@pytest.mark.unit
class TestSendHITLGmailNotificationHelper:
    """Tests for the convenience function send_hitl_gmail_notification."""

    @pytest.mark.asyncio
    async def test_returns_none_on_success(self):
        """Test that None is returned on successful send."""
        from seer.core.schema.models import GmailDeliveryConfig

        with patch("seer.services.workflows.hitl_email.HITLEmailService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.send_hitl_email = AsyncMock()
            mock_service_cls.return_value = mock_service

            from seer.services.workflows.hitl_email import send_hitl_gmail_notification

            result = await send_hitl_gmail_notification(
                user=MagicMock(),
                workflow_run=MagicMock(run_id="run_1"),
                interrupt_data={"type": "hitl"},
                gmail_config=GmailDeliveryConfig(recipient_email="test@example.com"),
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_error_message_on_failure(self):
        """Test that error message is returned on failure."""
        from seer.core.schema.models import GmailDeliveryConfig

        with patch("seer.services.workflows.hitl_email.HITLEmailService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.send_hitl_email = AsyncMock(side_effect=Exception("Gmail API error"))
            mock_service_cls.return_value = mock_service

            from seer.services.workflows.hitl_email import send_hitl_gmail_notification

            result = await send_hitl_gmail_notification(
                user=MagicMock(),
                workflow_run=MagicMock(run_id="run_2"),
                interrupt_data={"type": "hitl"},
                gmail_config=GmailDeliveryConfig(recipient_email="test@example.com"),
            )

            assert result == "Gmail API error"
