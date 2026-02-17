"""Browser profile models for persistent browser sessions."""
from __future__ import annotations

from tortoise import fields, models


class BrowserProfile(models.Model):
    """
    A browser profile containing saved login sessions.

    Like a Chrome profile - can have multiple sites logged in simultaneously.
    Users manage profiles explicitly via API, workflows reference them by ID.

    Session state includes cookies and localStorage for all logged-in domains,
    stored encrypted in session_state_enc.
    """

    id = fields.UUIDField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="browser_profiles")
    name = fields.CharField(max_length=100)  # e.g., "Work Profile", "Personal"

    # Encrypted Playwright storage_state JSON (cookies + localStorage)
    # Encryption handled at service layer before storage
    session_state_enc = fields.TextField(null=True)

    # Metadata about logged-in services (extracted from cookies)
    logged_in_domains = fields.JSONField(default=list)  # ["slack.com", "github.com"]

    status = fields.CharField(max_length=20, default="active")  # active, deleted
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    last_used_at = fields.DatetimeField(null=True)

    class Meta:
        table = "browser_profiles"
        unique_together = (("user", "name"),)

    def __str__(self) -> str:
        domains = ", ".join(self.logged_in_domains[:3]) if self.logged_in_domains else "no logins"
        if len(self.logged_in_domains) > 3:
            domains += f" +{len(self.logged_in_domains) - 3} more"
        return f"{self.name} ({domains})"
