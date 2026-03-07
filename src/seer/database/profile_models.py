import re

from tortoise import fields, models

RESERVED_USERNAMES = frozenset({
    "admin", "api", "explore", "settings", "login", "signup",
    "about", "help", "support", "pricing", "blog", "docs",
    "templates", "workflows", "public", "static", "assets",
    "u", "user", "users", "seer", "null", "undefined",
})

USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,28}[a-z0-9]$")


def validate_username(username: str) -> str:
    username = username.lower().strip()
    if not USERNAME_PATTERN.match(username):
        raise ValueError(
            "Username must be 3-30 chars, lowercase alphanumeric and hyphens, "
            "cannot start or end with a hyphen"
        )
    if username in RESERVED_USERNAMES:
        raise ValueError(f"Username '{username}' is reserved")
    return username


class UserProfile(models.Model):
    id = fields.IntField(primary_key=True)
    user = fields.OneToOneField(
        "models.User", related_name="profile", on_delete=fields.CASCADE
    )
    username = fields.CharField(max_length=30, unique=True, db_index=True)
    display_name = fields.CharField(max_length=100, null=True)
    bio = fields.TextField(null=True)
    avatar_url = fields.CharField(max_length=500, null=True)
    social_links = fields.JSONField(default=dict)  # {twitter, linkedin, website}
    tags = fields.JSONField(default=list)  # ["productivity", "marketing"]
    is_public = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "user_profiles"
