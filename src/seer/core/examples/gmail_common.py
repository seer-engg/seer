"""
Shared Gmail helper utilities for workflow examples.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, TypedDict

from tortoise import Tortoise

from seer.api.integrations.services import get_valid_access_token

logger = logging.getLogger(__name__)

USE_REAL_GMAIL = os.getenv("USE_REAL_GMAIL", "1").lower() in {"1", "true", "yes"}
print(f"USE_REAL_GMAIL: {USE_REAL_GMAIL}")

try:
    from seer.database.config import TORTOISE_ORM
except RuntimeError as exc:  # pragma: no cover - depends on env
    logger.warning("DATABASE_URL not configured (%s); DB lookups will be skipped", exc)
    TORTOISE_ORM = None

try:
    from seer.database.models import User
    from seer.database.models_oauth import OAuthConnection
except RuntimeError as exc:  # pragma: no cover - depends on env
    logger.warning("Unable to import OAuthConnection (%s); DB lookups will be skipped", exc)
    OAuthConnection = None  # type: ignore[assignment]
    User = None  # type: ignore[assignment]

from seer.tools.google.gmail import GmailCreateDraftTool, GmailReadTool  # noqa: E402


class GmailCredentials(TypedDict, total=False):
    access_token: str
    refresh_token: Optional[str]
    provider: str


async def fetch_oauth_credentials(user_id: int, provider: str = "google") -> Optional[GmailCredentials]:
    """
    Fetch OAuth credentials for demo purposes. Returns None when credentials are
    unavailable so callers can fall back to stubbed Gmail data.
    """

    if TORTOISE_ORM is None or OAuthConnection is None:
        logger.info("Skipping OAuth lookup because DATABASE_URL is not configured")
        return None

    try:  # pragma: no cover - requires DB connectivity
        await Tortoise.init(config=TORTOISE_ORM)
        conn = await OAuthConnection.filter(user_id=user_id, provider=provider).first()
        if conn:
            logger.info("Loaded OAuth credentials for user_id=%s provider=%s", user_id, provider)
            user = await User.get(id=user_id)
            access_token = await get_valid_access_token(user, provider)
            if access_token:
                return GmailCredentials(
                    access_token=access_token,
                    refresh_token=conn.refresh_token_enc,
                    provider=provider,
                )
            else:
                logger.error("No valid access token found for user_id=%s provider=%s", user_id, provider)
                return None
    except Exception as exc:  # pragma: no cover - best-effort helper
        logger.warning("Unable to fetch OAuth credentials, continuing with stub data: %s", exc)
    finally:
        await Tortoise.close_connections()
    return None


class GmailDemoService:
    """
    Lightweight adapter that wraps Gmail tool invocations, falling back to stub
    data when credentials/environment are not available.
    """

    def __init__(self, credentials: Optional[GmailCredentials]) -> None:
        self.credentials = credentials
        self.read_tool = GmailReadTool()
        self.create_draft_tool = GmailCreateDraftTool()

    def read_emails(self, params: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if USE_REAL_GMAIL and self.credentials:
            logger.info("Fetching Gmail messages via API")
            return asyncio.run(  # pragma: no cover - requires live credentials
                self.read_tool.execute(self.credentials["access_token"], dict(params))
            )

        logger.info("Using stub Gmail inbox data")
        max_results = int(params.get("max_results", 1))
        sample = {
            "id": "demo-message-1",
            "threadId": "demo-thread",
            "snippet": "Reminder about tomorrow's demo",
            "subject": "Demo tomorrow?",
            "from": "product@example.com",
            "to": "you@example.com",
            "date": "Fri, 13 Dec 2025 10:00:00 -0000",
            "labelIds": ["INBOX"],
            "body": (
                "Hey there,\n\nCan we confirm the talking points for tomorrow's demo?\n"
                "Let me know if you need anything else.\n\n- Product"
            ),
        }
        return [sample] * max(1, max_results)

    def create_draft(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        if USE_REAL_GMAIL and self.credentials:
            logger.info("Creating Gmail draft via API")
            payload = {
                "to": params.get("to", []),
                "subject": params.get("subject"),
                "body_text": params.get("body_text"),
                "body_html": params.get("body_html"),
            }
            return asyncio.run(  # pragma: no cover - requires live credentials
                self.create_draft_tool.execute(self.credentials["access_token"], payload)
            )

        logger.info("Stubbing Gmail draft creation")
        to_list = params.get("to") or []
        return {
            "id": "demo-draft-1",
            "message": {
                "id": "demo-message-2",
                "threadId": "demo-thread",
                "labelIds": ["DRAFT"],
                "snippet": (params.get("body_text") or "")[:140],
                "payload": {
                    "headers": [
                        {"name": "To", "value": ", ".join(to_list)},
                        {"name": "Subject", "value": params.get("subject", "")},
                    ],
                    "body": {"size": len(params.get("body_text", ""))},
                },
            },
        }
