"""
Email service for sending transactional emails.

This module provides a provider-agnostic email service for sending
transactional emails like organization invitations, approval notifications,
and member joined notifications.

Supported providers:
- Resend (default)
- SendGrid (planned)
- AWS SES (planned)

Usage:
    from seer.services.email_service import send_invitation_email

    await send_invitation_email(
        to_email="user@example.com",
        organization_name="Acme Corp",
        invited_by_name="John Doe",
        role="admin",
        invite_url="https://app.getseer.dev/invitations/abc123",
    )
"""
from typing import Any, Dict, List, Optional

import httpx

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)


class EmailServiceError(Exception):
    """Error sending email."""

    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


# =============================================================================
# Low-Level Email Sending
# =============================================================================


async def _send_via_resend(
    to_emails: List[str],
    subject: str,
    html_body: str,
    *,
    text_body: Optional[str] = None,
    from_address: Optional[str] = None,
    reply_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send email via Resend API.

    Resend API docs: https://resend.com/docs/api-reference/emails/send-email
    """
    api_key = config.email_api_key
    if not api_key:
        raise EmailServiceError("Resend API key not configured", provider="resend")

    from_addr = from_address or config.email_from_address

    payload = {
        "from": from_addr,
        "to": to_emails,
        "subject": subject,
        "html": html_body,
    }

    if text_body:
        payload["text"] = text_body

    if reply_to:
        payload["reply_to"] = reply_to

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        if not response.is_success:
            logger.error(
                "Resend API error: %s %s",
                response.status_code,
                response.text[:200],
            )
            raise EmailServiceError(
                f"Failed to send email: {response.status_code}",
                provider="resend",
                status_code=response.status_code,
            )

        return response.json()


async def send_email(  # pylint: disable=too-many-arguments  # Reason: email params + tracking params
    to_emails: List[str],
    subject: str,
    html_body: str,
    *,
    text_body: Optional[str] = None,
    from_address: Optional[str] = None,
    reply_to: Optional[str] = None,
    email_type: str = "transactional",
    organization_id: Optional[int] = None,
    user_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Send an email using the configured provider with tracking.

    Automatically injects a tracking pixel and rewrites links for
    open/click analytics before sending.

    Args:
        to_emails: List of recipient email addresses
        subject: Email subject line
        html_body: HTML body content
        text_body: Optional plain text body (fallback)
        from_address: Optional override for from address
        reply_to: Optional reply-to address
        email_type: Type of email for analytics (e.g. "invitation", "approval")
        organization_id: Optional org ID for analytics correlation
        user_id: Optional user ID for analytics correlation

    Returns:
        Provider-specific response data, or None if email is disabled

    Raises:
        EmailServiceError: If sending fails
    """
    from seer.services.email_tracking import (  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import at module level
        create_tracking_record,
        finalize_send,
        inject_tracking,
    )

    provider = str(config.email_provider).lower()

    if provider == "disabled":
        logger.info("Email sending disabled, skipping email to %s", to_emails)
        return None

    # Create tracking record and inject tracking into HTML
    primary_recipient = to_emails[0] if to_emails else ""
    tracking_id = await create_tracking_record(
        provider=provider,
        email_type=email_type,
        recipient=primary_recipient,
        subject=subject,
        organization_id=organization_id,
        user_id=user_id,
    )
    tracked_html = inject_tracking(html_body, tracking_id)

    if provider == "resend":
        result = await _send_via_resend(
            to_emails=to_emails,
            subject=subject,
            html_body=tracked_html,
            text_body=text_body,
            from_address=from_address,
            reply_to=reply_to,
        )
        await finalize_send(tracking_id, provider_email_id=result.get("id"))
        return result

    # Future: Add SendGrid, AWS SES support
    raise EmailServiceError(f"Unsupported email provider: {provider}", provider=provider)


# =============================================================================
# Email Templates
# =============================================================================


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _render_email_layout(title: str, content: str, footer_text: str = "") -> str:
    """Render base email layout with consistent styling."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f3f4f6;">
    <div style="border: 1px solid #e5e7eb; border-radius: 12px; padding: 32px; background-color: #ffffff;">
        <div style="text-align: center; margin-bottom: 24px;">
            <span style="font-size: 24px; font-weight: 700; color: #111827;">Seer</span>
        </div>

        <h1 style="margin: 0 0 16px 0; color: #111827; font-size: 22px; font-weight: 600; text-align: center;">
            {_escape_html(title)}
        </h1>

        {content}
    </div>

    <div style="margin-top: 24px; padding-top: 16px;">
        <p style="color: #9ca3af; font-size: 12px; text-align: center; margin: 0;">
            {_escape_html(footer_text) if footer_text else "This is an automated message from Seer."}
        </p>
    </div>
</body>
</html>
    """.strip()


def _render_cta_button(url: str, text: str) -> str:
    """Render a call-to-action button."""
    return f"""
<div style="text-align: center; margin: 32px 0;">
    <a href="{url}"
       style="display: inline-block; background-color: #2563eb; color: #ffffff; padding: 14px 32px; text-decoration: none; border-radius: 8px; font-size: 16px; font-weight: 500; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
        {_escape_html(text)}
    </a>
</div>

<p style="color: #9ca3af; font-size: 13px; text-align: center; margin: 24px 0 0 0;">
    Or copy this link: <a href="{url}" style="color: #6b7280;">{url}</a>
</p>
    """.strip()


# =============================================================================
# High-Level Email Functions
# =============================================================================


async def send_invitation_email(
    to_email: str,
    organization_name: str,
    invited_by_name: str,
    role: str,
    invite_url: str,
) -> Optional[Dict[str, Any]]:
    """
    Send a team invitation email.

    Args:
        to_email: Recipient email address
        organization_name: Name of the organization
        invited_by_name: Name of the person who sent the invite
        role: Role being offered (admin, user, consultant)
        invite_url: URL to accept the invitation

    Returns:
        Provider response or None if email is disabled
    """
    title = f"You've been invited to join {organization_name}"

    content = f"""
<p style="color: #4b5563; font-size: 15px; line-height: 1.6; margin: 16px 0; text-align: center;">
    <strong>{_escape_html(invited_by_name)}</strong> has invited you to join
    <strong>{_escape_html(organization_name)}</strong> on Seer as a <strong>{_escape_html(role)}</strong>.
</p>

<p style="color: #4b5563; font-size: 15px; line-height: 1.6; margin: 16px 0; text-align: center;">
    Click the button below to accept the invitation and get started.
</p>

{_render_cta_button(invite_url, "Accept Invitation")}

<p style="color: #6b7280; font-size: 14px; margin: 24px 0 0 0; text-align: center;">
    This invitation will expire in 7 days.
</p>
    """

    html_body = _render_email_layout(title, content)

    text_body = f"""
You've been invited to join {organization_name}

{invited_by_name} has invited you to join {organization_name} on Seer as a {role}.

Click the link below to accept the invitation:
{invite_url}

This invitation will expire in 7 days.

---
This is an automated message from Seer.
    """.strip()

    logger.info(
        "Sending invitation email to %s for org %s",
        to_email,
        organization_name,
    )

    return await send_email(
        to_emails=[to_email],
        subject=f"You've been invited to join {organization_name} on Seer",
        html_body=html_body,
        text_body=text_body,
        email_type="invitation",
    )


async def send_approval_notification_email(
    to_emails: List[str],
    workflow_name: str,
    requested_by_name: str,
    organization_name: str,
    review_url: str,
) -> Optional[Dict[str, Any]]:
    """
    Notify admins/owners about a pending workflow approval request.

    Args:
        to_emails: List of admin/owner email addresses
        workflow_name: Name of the workflow requiring approval
        requested_by_name: Name of the consultant who created the workflow
        organization_name: Name of the organization
        review_url: URL to review the workflow

    Returns:
        Provider response or None if email is disabled
    """
    title = f"Workflow approval requested: {workflow_name}"

    content = f"""
<p style="color: #4b5563; font-size: 15px; line-height: 1.6; margin: 16px 0; text-align: center;">
    <strong>{_escape_html(requested_by_name)}</strong> has submitted a workflow for approval
    in <strong>{_escape_html(organization_name)}</strong>.
</p>

<div style="background-color: #f8f9fa; border-radius: 8px; padding: 16px; margin: 20px 0;">
    <p style="margin: 0; color: #374151; font-size: 14px;">
        <strong>Workflow:</strong> {_escape_html(workflow_name)}
    </p>
</div>

<p style="color: #4b5563; font-size: 15px; line-height: 1.6; margin: 16px 0; text-align: center;">
    Please review and approve or reject this workflow.
</p>

{_render_cta_button(review_url, "Review Workflow")}
    """

    html_body = _render_email_layout(title, content)

    text_body = f"""
Workflow approval requested: {workflow_name}

{requested_by_name} has submitted a workflow for approval in {organization_name}.

Workflow: {workflow_name}

Please review and approve or reject this workflow:
{review_url}

---
This is an automated message from Seer.
    """.strip()

    logger.info(
        "Sending approval notification for workflow %s to %d recipients",
        workflow_name,
        len(to_emails),
    )

    return await send_email(
        to_emails=to_emails,
        subject=f"Workflow approval requested: {workflow_name}",
        html_body=html_body,
        text_body=text_body,
        email_type="approval",
    )


async def send_member_joined_notification(
    to_email: str,
    new_member_name: str,
    new_member_email: str,
    organization_name: str,
    role: str,
) -> Optional[Dict[str, Any]]:
    """
    Notify owner when a new member joins the organization.

    Args:
        to_email: Owner's email address
        new_member_name: Name of the new member
        new_member_email: Email of the new member
        organization_name: Name of the organization
        role: Role assigned to the new member

    Returns:
        Provider response or None if email is disabled
    """
    title = f"{new_member_name} joined {organization_name}"

    member_display = new_member_name if new_member_name else new_member_email

    content = f"""
<p style="color: #4b5563; font-size: 15px; line-height: 1.6; margin: 16px 0; text-align: center;">
    <strong>{_escape_html(member_display)}</strong> has joined your organization
    <strong>{_escape_html(organization_name)}</strong> as a <strong>{_escape_html(role)}</strong>.
</p>

<div style="background-color: #f8f9fa; border-radius: 8px; padding: 16px; margin: 20px 0; text-align: center;">
    <p style="margin: 0 0 8px 0; color: #374151; font-size: 14px;">
        <strong>Member:</strong> {_escape_html(member_display)}
    </p>
    <p style="margin: 0 0 8px 0; color: #374151; font-size: 14px;">
        <strong>Email:</strong> {_escape_html(new_member_email)}
    </p>
    <p style="margin: 0; color: #374151; font-size: 14px;">
        <strong>Role:</strong> {_escape_html(role)}
    </p>
</div>
    """

    html_body = _render_email_layout(title, content)

    text_body = f"""
{new_member_name} joined {organization_name}

{member_display} has joined your organization {organization_name} as a {role}.

Member: {member_display}
Email: {new_member_email}
Role: {role}

---
This is an automated message from Seer.
    """.strip()

    logger.info(
        "Sending member joined notification for %s to owner %s",
        new_member_email,
        to_email,
    )

    return await send_email(
        to_emails=[to_email],
        subject=f"{new_member_name or new_member_email} joined {organization_name}",
        html_body=html_body,
        text_body=text_body,
        email_type="member_joined",
    )


async def send_invitation_accepted_notification(
    to_email: str,
    member_name: str,
    member_email: str,
    organization_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Notify the inviter when their invitation is accepted.

    Args:
        to_email: Inviter's email address
        member_name: Name of the person who accepted
        member_email: Email of the person who accepted
        organization_name: Name of the organization

    Returns:
        Provider response or None if email is disabled
    """
    return await send_member_joined_notification(
        to_email=to_email,
        new_member_name=member_name,
        new_member_email=member_email,
        organization_name=organization_name,
        role="member",  # The actual role would be in the invitation
    )
