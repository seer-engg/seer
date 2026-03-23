"""
HITL Email Service - Sends HITL notification emails with form links.

This service handles the Gmail delivery channel for HITL nodes by:
1. Creating a temporary HITL form via HITLFormService
2. Rendering an HTML email with the form link
3. Sending the email via the user's Gmail OAuth connection
"""

from typing import Any, Dict, List, Optional

from seer.core.schema.models import GmailDeliveryConfig
from seer.database import User, WorkflowRun
from seer.logger import get_logger
from seer.services.workflows.hitl_form import HITLFormService
from seer.tools.google.gmail.send import GmailSendEmailTool
from seer.tools.oauth_manager import get_oauth_token

logger = get_logger(__name__)


class HITLEmailService:
    """
    Service for sending HITL notification emails via Gmail.

    Emails contain a prominent call-to-action button linking to the HITL form.
    When the user clicks the link and submits the form, the workflow resumes.
    """

    def __init__(self) -> None:
        self._form_service = HITLFormService()
        self._gmail_tool = GmailSendEmailTool()

    async def send_hitl_email(
        self,
        user: User,
        workflow_run: WorkflowRun,
        interrupt_data: Dict[str, Any],
        gmail_config: GmailDeliveryConfig,
    ) -> None:
        """
        Send an HITL notification email with a form link.

        Args:
            user: The workflow owner (used to resolve Gmail OAuth connection)
            workflow_run: The interrupted workflow run
            interrupt_data: The HITL interrupt payload
            gmail_config: Gmail delivery configuration (recipient email)

        Raises:
            HTTPException: If Gmail connection is not available or email fails to send
        """
        # Create HITL form
        _, form_url = await self._form_service.create_hitl_form(
            workflow_run, interrupt_data
        )

        # Render email content
        title = interrupt_data.get("title", "Action Required")
        subject = f"[Action Required] {title}"
        body_html = self._render_email_html(interrupt_data, form_url)
        body_text = self._render_email_text(interrupt_data, form_url)

        # Inject email tracking
        tracking_id = None
        try:
            from seer.services.email_tracking import create_tracking_record, inject_tracking  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import at module level

            tracking_id = await create_tracking_record(
                provider="gmail",
                email_type="hitl",
                recipient=gmail_config.recipient_email,
                subject=subject,
                workflow_run_id=workflow_run.run_id,
                user_id=user.user_id,
            )
            body_html = inject_tracking(body_html, tracking_id)
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: tracking failures must not block HITL email
            logger.warning("Failed to set up HITL email tracking for run '%s'", workflow_run.run_id)
            tracking_id = None

        # Get Gmail OAuth connection (auto-select user's Google connection)
        connection, access_token = await get_oauth_token(
            user,
            provider="google",  # Gmail uses Google OAuth
        )

        logger.info(
            "Sending HITL email for run '%s' to '%s'",
            workflow_run.run_id,
            gmail_config.recipient_email,
            extra={
                "run_id": workflow_run.run_id,
                "recipient": gmail_config.recipient_email,
                "connection_id": connection.id,
                "form_url": form_url,
            },
        )

        # Send via Gmail API
        result = await self._gmail_tool.execute(
            access_token=access_token,
            arguments={
                "to": [gmail_config.recipient_email],
                "subject": subject,
                "body_text": body_text,
                "body_html": body_html,
            },
        )

        # Finalize tracking
        if tracking_id:
            try:
                from seer.services.email_tracking import finalize_send  # pylint: disable=import-outside-toplevel  # Reason: lazy import to match tracking setup above
                await finalize_send(tracking_id, provider_email_id=result.get("id"))
            except Exception:  # pylint: disable=broad-exception-caught  # Reason: tracking failures must not affect HITL flow
                logger.warning("Failed to finalize HITL email tracking for run '%s'", workflow_run.run_id)

        logger.info(
            "HITL email sent successfully for run '%s'",
            workflow_run.run_id,
            extra={"run_id": workflow_run.run_id},
        )

    def _render_email_html(self, data: Dict[str, Any], form_url: str) -> str:
        """
        Render HTML email content with HITL form link.

        Uses inline styles since email clients don't support external CSS.
        The prominent CTA button encourages users to click through to the form.
        """
        title = data.get("title", "Action Required")
        description = data.get("description")
        display_items: List[Dict[str, Any]] = data.get("display", [])

        # Build display section HTML
        display_html = ""
        if display_items:
            display_html = """
            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 16px; margin: 20px 0;">
                <h3 style="margin: 0 0 12px 0; color: #374151; font-size: 14px; font-weight: 600;">
                    Information
                </h3>
                <table style="width: 100%; border-collapse: collapse;">
            """
            for item in display_items:
                label = self._escape_html(str(item.get("label", "")))
                value = self._escape_html(str(item.get("value", "")))
                display_html += f"""
                    <tr>
                        <td style="padding: 4px 8px 4px 0; color: #6b7280; font-size: 14px; vertical-align: top;">
                            {label}:
                        </td>
                        <td style="padding: 4px 0; color: #111827; font-size: 14px;">
                            {value}
                        </td>
                    </tr>
                """
            display_html += """
                </table>
            </div>
            """

        # Build description section
        description_html = ""
        if description:
            description_html = f"""
            <p style="color: #4b5563; font-size: 15px; line-height: 1.6; margin: 16px 0;">
                {self._escape_html(description)}
            </p>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #ffffff;">
    <div style="border: 1px solid #e5e7eb; border-radius: 12px; padding: 32px; background-color: #ffffff;">
        <h1 style="margin: 0 0 16px 0; color: #111827; font-size: 24px; font-weight: 600;">
            {self._escape_html(title)}
        </h1>

        {description_html}
        {display_html}

        <p style="color: #4b5563; font-size: 15px; margin: 24px 0 16px 0;">
            Please click the button below to provide your response:
        </p>

        <div style="text-align: center; margin: 32px 0;">
            <a href="{form_url}"
               style="display: inline-block; background-color: #2563eb; color: #ffffff; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-size: 16px; font-weight: 500; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
                Respond Now
            </a>
        </div>

        <p style="color: #9ca3af; font-size: 13px; margin: 24px 0 0 0;">
            Or copy this link: <a href="{form_url}" style="color: #6b7280;">{form_url}</a>
        </p>
    </div>

    <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #e5e7eb;">
        <p style="color: #9ca3af; font-size: 12px; text-align: center; margin: 0;">
            This is an automated message from Seer Workflow.
        </p>
    </div>
</body>
</html>
        """.strip()

    def _render_email_text(self, data: Dict[str, Any], form_url: str) -> str:
        """
        Render plain text email content as fallback.
        """
        title = data.get("title", "Action Required")
        description = data.get("description", "")
        display_items: List[Dict[str, Any]] = data.get("display", [])

        lines = [
            title,
            "=" * len(title),
            "",
        ]

        if description:
            lines.extend([description, ""])

        if display_items:
            lines.append("Information:")
            for item in display_items:
                label = item.get("label", "")
                value = item.get("value", "")
                lines.append(f"  - {label}: {value}")
            lines.append("")

        lines.extend([
            "Please click the link below to provide your response:",
            "",
            form_url,
            "",
            "---",
            "This is an automated message from Seer Workflow.",
        ])

        return "\n".join(lines)

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters to prevent XSS."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )


async def send_hitl_gmail_notification(
    user: User,
    workflow_run: WorkflowRun,
    interrupt_data: Dict[str, Any],
    gmail_config: GmailDeliveryConfig,
) -> Optional[str]:
    """
    Convenience function to send an HITL Gmail notification.

    Returns:
        Error message if sending failed, None on success
    """
    service = HITLEmailService()
    try:
        await service.send_hitl_email(user, workflow_run, interrupt_data, gmail_config)
        return None
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Intentional: convert any error to string for caller
        logger.exception(
            "Failed to send HITL email for run '%s'",
            workflow_run.run_id,
            extra={"run_id": workflow_run.run_id, "error": str(exc)},
        )
        return str(exc)
