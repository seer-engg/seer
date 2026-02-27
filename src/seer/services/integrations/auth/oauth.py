import os

from authlib.integrations.starlette_client import OAuth

from seer.config import config
from seer.services.integrations.constants import SUPABASE_OAUTH_PROVIDER

oauth = OAuth()

# Google
# Scopes are controlled by frontend - minimal default for identity only
oauth.register(
    name='google',
    client_id=config.google_client_id,
    client_secret=config.google_client_secret,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'  # Minimal default - frontend will override with specific scopes
    }
)

# GitHub
# Scopes are controlled by frontend - minimal default for identity only
oauth.register(
    name='github',
    client_id=config.github_client_id or os.getenv('GITHUB_CLIENT_ID'),
    client_secret=config.github_client_secret or os.getenv('GITHUB_CLIENT_SECRET'),
    authorize_url='https://github.com/login/oauth/authorize',
    access_token_url='https://github.com/login/oauth/access_token',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},  # Minimal default - frontend will override with specific scopes
)


def _supabase_base() -> str:
    base = config.supabase_management_api_base or "https://api.supabase.com"
    return base.rstrip("/")


if config.supabase_client_id and config.supabase_client_secret:
    oauth.register(
        name='supabase_mgmt',
        client_id=config.supabase_client_id,
        client_secret=config.supabase_client_secret,
        authorize_url=f"{_supabase_base()}/v1/oauth/authorize",
        access_token_url=f"{_supabase_base()}/v1/oauth/token",
        api_base_url=f"{_supabase_base()}/",
        client_kwargs={'scope': 'read:projects'},
    )

# Discord Bot Installation
if config.discord_client_id and config.discord_client_secret:
    oauth.register(
        name='discord',
        client_id=config.discord_client_id,
        client_secret=config.discord_client_secret,
        authorize_url='https://discord.com/api/oauth2/authorize',
        access_token_url='https://discord.com/api/oauth2/token',
        api_base_url='https://discord.com/api/',
        client_kwargs={'scope': 'bot'},
    )

# LinkedIn OAuth
# Scopes are controlled by frontend - minimal default for identity only
if config.linkedin_client_id and config.linkedin_client_secret:
    oauth.register(
        name='linkedin',
        client_id=config.linkedin_client_id,
        client_secret=config.linkedin_client_secret,
        authorize_url='https://www.linkedin.com/oauth/v2/authorization',
        access_token_url='https://www.linkedin.com/oauth/v2/accessToken',
        api_base_url='https://api.linkedin.com/',
        client_kwargs={'scope': 'openid profile email'},  # Minimal default - frontend will override with specific scopes
    )

# Slack OAuth (Bot Token)
# Slack uses OAuth 2.0 v2 with bot token scopes
if config.slack_client_id and config.slack_client_secret:
    oauth.register(
        name='slack',
        client_id=config.slack_client_id,
        client_secret=config.slack_client_secret,
        authorize_url='https://slack.com/oauth/v2/authorize',
        access_token_url='https://slack.com/api/oauth.v2.access',
        api_base_url='https://slack.com/api/',
        client_kwargs={'scope': 'channels:read chat:write'},  # Minimal default - frontend will override with specific scopes
    )

# Airtable OAuth (requires PKCE)
# Airtable uses OAuth 2.0 with PKCE (Proof Key for Code Exchange)
# Important: Airtable requires Basic Auth for token endpoint
# Authlib handles PKCE automatically when code_challenge_method is set
if config.airtable_client_id and config.airtable_client_secret:
    oauth.register(
        name='airtable',
        client_id=config.airtable_client_id,
        client_secret=config.airtable_client_secret,
        authorize_url='https://airtable.com/oauth2/v1/authorize',
        access_token_url='https://airtable.com/oauth2/v1/token',
        api_base_url='https://api.airtable.com/',
        token_endpoint_auth_method='client_secret_basic',  # Required: Airtable needs Basic Auth header
        client_kwargs={
            'scope': 'data.records:read schema.bases:read',
            'code_challenge_method': 'S256',  # Authlib handles PKCE automatically
        },
    )


# Provider mappings: integration_type -> OAuth provider
_INTEGRATION_TO_PROVIDER: dict[str, str] = {
    # Google integrations
    'gmail': 'google',
    'googlesheets': 'google',
    'googledrive': 'google',
    'google': 'google',
    'google_sheets': 'google',
    'google_drive': 'google',
    'google_calendar': 'google',
    # GitHub integrations
    'github': 'github',
    'pull_request': 'github',
    # Supabase integrations
    'supabase': SUPABASE_OAUTH_PROVIDER,
    'supabase_mgmt': SUPABASE_OAUTH_PROVIDER,
    # Direct providers
    'discord': 'discord',
    'linkedin': 'linkedin',
    'slack': 'slack',
    'airtable': 'airtable',
}


def get_oauth_provider(integration_type: str) -> str:
    """
    Map integration type to OAuth provider.
    Multiple integration types can share the same OAuth provider.

    Args:
        integration_type: Integration type (gmail, google_sheets, google_drive, etc.)

    Returns:
        OAuth provider name (google, github, etc.)
    """
    # For unmapped providers, the integration type is the same as the provider
    return _INTEGRATION_TO_PROVIDER.get(integration_type, integration_type)
