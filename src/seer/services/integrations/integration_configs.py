"""
Integration configuration data.

Static configuration for all supported integrations.
When adding a new integration, add its configuration here.
"""
from typing import Any, Dict


INTEGRATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Google integrations
    "gmail": {
        "display_name": "Gmail",
        "oauth_provider": "google",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/gmail.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#EA4335",
        "default_scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
        "scopes": [
            {
                "value": "https://www.googleapis.com/auth/gmail.readonly",
                "display_name": "Read emails",
                "description": "Read email messages and labels"
            },
            {
                "value": "https://www.googleapis.com/auth/gmail.send",
                "display_name": "Send emails",
                "description": "Send email messages on your behalf"
            },
            {
                "value": "https://www.googleapis.com/auth/gmail.compose",
                "display_name": "Compose drafts",
                "description": "Create and modify email drafts"
            },
            {
                "value": "https://www.googleapis.com/auth/gmail.modify",
                "display_name": "Modify emails",
                "description": "Modify email labels and read/unread status"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["gmail_"],
            "scope_keywords": ["gmail"]
        }
    },
    "google_drive": {
        "display_name": "Google Drive",
        "oauth_provider": "google",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/drive.google.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#4285F4",
        "default_scopes": ["https://www.googleapis.com/auth/drive.readonly"],
        "scopes": [
            {
                "value": "https://www.googleapis.com/auth/drive.readonly",
                "display_name": "Read files",
                "description": "View files and folders in Google Drive"
            },
            {
                "value": "https://www.googleapis.com/auth/drive.file",
                "display_name": "Manage files",
                "description": "Create, edit, and delete files created by this app"
            },
            {
                "value": "https://www.googleapis.com/auth/drive",
                "display_name": "Full access",
                "description": "Full read/write access to all Google Drive files"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["gdrive_", "google_drive_"],
            "scope_keywords": ["drive"]
        }
    },
    "google_sheets": {
        "display_name": "Google Sheets",
        "oauth_provider": "google",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/sheets.google.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#0F9D58",
        "default_scopes": ["https://www.googleapis.com/auth/spreadsheets.readonly"],
        "scopes": [
            {
                "value": "https://www.googleapis.com/auth/spreadsheets.readonly",
                "display_name": "Read spreadsheets",
                "description": "View spreadsheets and their data"
            },
            {
                "value": "https://www.googleapis.com/auth/spreadsheets",
                "display_name": "Edit spreadsheets",
                "description": "Read and write spreadsheet data"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["gsheets_", "google_sheets_"],
            "scope_keywords": ["spreadsheets"]
        }
    },
    "google_calendar": {
        "display_name": "Google Calendar",
        "oauth_provider": "google",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/calendar.google.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#4285F4",
        "default_scopes": ["https://www.googleapis.com/auth/calendar.readonly"],
        "scopes": [
            {
                "value": "https://www.googleapis.com/auth/calendar.readonly",
                "display_name": "Read calendars",
                "description": "View calendar events and settings"
            },
            {
                "value": "https://www.googleapis.com/auth/calendar.events",
                "display_name": "Manage events",
                "description": "Create, edit, and delete calendar events"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["google_calendar_", "gcalendar_"],
            "scope_keywords": ["calendar"]
        }
    },
    "youtube": {
        "display_name": "YouTube",
        "oauth_provider": "google",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/youtube.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#FF0000",
        "default_scopes": ["https://www.googleapis.com/auth/youtube.readonly"],
        "scopes": [
            {
                "value": "https://www.googleapis.com/auth/youtube.readonly",
                "display_name": "View YouTube data",
                "description": "Search videos, view channel info, and list playlists"
            },
            {
                "value": "https://www.googleapis.com/auth/youtube.upload",
                "display_name": "Upload videos",
                "description": "Upload videos to your YouTube channel"
            },
            {
                "value": "https://www.googleapis.com/auth/youtube",
                "display_name": "Manage YouTube",
                "description": "Full access to manage your YouTube account"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["youtube_"],
            "scope_keywords": ["youtube"]
        }
    },
    # GitHub integrations
    "github": {
        "display_name": "GitHub",
        "oauth_provider": "github",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/github.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#24292F",
        "default_scopes": ["repo"],
        "scopes": [
            {
                "value": "repo",
                "display_name": "Repository access",
                "description": "Full control of private repositories"
            },
            {
                "value": "user:email",
                "display_name": "Email access",
                "description": "Access email addresses (read-only)"
            },
            {
                "value": "read:user",
                "display_name": "Profile access",
                "description": "Read user profile data"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["github_"],
            "scope_keywords": ["repo", "github"]
        }
    },
    "pull_request": {
        "display_name": "GitHub Pull Requests",
        "oauth_provider": "github",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/github.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#24292F",
        "default_scopes": ["repo"],
        "scopes": [
            {
                "value": "repo",
                "display_name": "Repository access",
                "description": "Full control of private repositories"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["github_pull_request", "github_list_pull"],
            "scope_keywords": ["repo"]
        }
    },
    # LinkedIn integration
    "linkedin": {
        "display_name": "LinkedIn",
        "oauth_provider": "linkedin",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/linkedin.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#0A66C2",
        "default_scopes": ["openid", "profile", "email"],
        "scopes": [
            {
                "value": "openid",
                "display_name": "OpenID",
                "description": "Authenticate with OpenID Connect"
            },
            {
                "value": "profile",
                "display_name": "Profile",
                "description": "Access your LinkedIn profile"
            },
            {
                "value": "email",
                "display_name": "Email",
                "description": "Access your email address"
            },
            {
                "value": "w_member_social",
                "display_name": "Post content",
                "description": "Create and share posts on your behalf"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["linkedin_"],
            "scope_keywords": ["w_member_social", "linkedin"]
        }
    },
    # Oura Ring integration
    "oura": {
        "display_name": "Oura Ring",
        "oauth_provider": "oura",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/ouraring.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#1A1A2E",
        "default_scopes": ["daily", "heartrate", "spo2", "stress", "session", "personal"],
        "scopes": [
            {"value": "daily", "display_name": "Daily summaries", "description": "Sleep, readiness, and activity scores"},
            {"value": "heartrate", "display_name": "Heart rate", "description": "Heart rate and HRV data"},
            {"value": "spo2", "display_name": "Blood oxygen", "description": "SpO2 readings"},
            {"value": "stress", "display_name": "Stress", "description": "Stress level data"},
            {"value": "workout", "display_name": "Workouts", "description": "Workout sessions"},
            {"value": "session", "display_name": "Sessions", "description": "Meditation and breathing sessions"},
            {"value": "tag", "display_name": "Tags", "description": "User-created tags"},
            {"value": "personal", "display_name": "Personal info", "description": "Profile information"},
        ],
        "detection_patterns": {
            "tool_name_patterns": ["oura_"],
            "scope_keywords": ["oura", "health", "sleep", "heart_rate"]
        }
    },
    # Discord integration
    "discord": {
        "display_name": "Discord",
        "oauth_provider": "discord",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/discord.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#5865F2",
        "default_scopes": ["bot"],
        "scopes": [
            {
                "value": "bot",
                "display_name": "Bot access",
                "description": "Install bot to your Discord server"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["discord_"],
            "scope_keywords": ["discord"]
        }
    },
    # Supabase integration
    "supabase": {
        "display_name": "Supabase",
        "oauth_provider": "supabase_mgmt",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/supabase.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#3ECF8E",
        "default_scopes": ["read:projects"],
        "scopes": [
            {
                "value": "read:projects",
                "display_name": "Read projects",
                "description": "View your Supabase projects"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["supabase_"],
            "scope_keywords": ["supabase"]
        }
    },
    # Slack integration
    "slack": {
        "display_name": "Slack",
        "oauth_provider": "slack",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/slack.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#4A154B",
        "default_scopes": ["channels:read", "chat:write"],
        "scopes": [
            {
                "value": "channels:read",
                "display_name": "View channels",
                "description": "View basic information about public channels in a workspace"
            },
            {
                "value": "channels:history",
                "display_name": "View channel messages",
                "description": "View messages and other content in public channels"
            },
            {
                "value": "chat:write",
                "display_name": "Send messages",
                "description": "Send messages as the app"
            },
            {
                "value": "channels:join",
                "display_name": "Join channels",
                "description": "Join public channels in the workspace"
            },
            {
                "value": "users:read",
                "display_name": "View users",
                "description": "View people in a workspace"
            },
            {
                "value": "groups:read",
                "display_name": "View private channels",
                "description": "View basic information about private channels"
            },
            {
                "value": "groups:history",
                "display_name": "View private channel messages",
                "description": "View messages and other content in private channels"
            },
            {
                "value": "im:write",
                "display_name": "Send direct messages",
                "description": "Start direct messages with people"
            },
            {
                "value": "reactions:write",
                "display_name": "Add reactions",
                "description": "Add emoji reactions to messages"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["slack_"],
            "scope_keywords": ["slack", "channels", "chat"]
        }
    },
    # Notion integration
    "notion": {
        "display_name": "Notion",
        "oauth_provider": "notion",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/notion.so?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#000000",
        "default_scopes": [],
        "scopes": [],
        "detection_patterns": {
            "tool_name_patterns": ["notion_"],
            "scope_keywords": ["notion"]
        }
    },
    # Airtable integration
    "airtable": {
        "display_name": "Airtable",
        "oauth_provider": "airtable",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/airtable.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#FCBC2E",
        "default_scopes": ["data.records:read", "schema.bases:read"],
        "scopes": [
            {
                "value": "data.records:read",
                "display_name": "Read records",
                "description": "Read records from tables in your bases"
            },
            {
                "value": "data.records:write",
                "display_name": "Write records",
                "description": "Create, update, and delete records in your bases"
            },
            {
                "value": "schema.bases:read",
                "display_name": "Read base schema",
                "description": "View base and table structure including field definitions"
            },
            {
                "value": "schema.bases:write",
                "display_name": "Write base schema",
                "description": "Create and modify bases and tables"
            },
            {
                "value": "user.email:read",
                "display_name": "Read email",
                "description": "Access your account email address"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["airtable_"],
            "scope_keywords": ["airtable"]
        }
    },
}
