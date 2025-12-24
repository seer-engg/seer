from authlib.integrations.starlette_client import OAuth
import os
from shared.config import config

oauth = OAuth()

# Google
# Scopes are controlled by frontend - minimal default for identity only
oauth.register(
    name='google',
    client_id=config.GOOGLE_CLIENT_ID,
    client_secret=config.GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'  # Minimal default - frontend will override with specific scopes
    }
)

# GitHub
# Scopes are controlled by frontend - minimal default for identity only
oauth.register(
    name='github',
    client_id=config.GITHUB_CLIENT_ID or os.getenv('GITHUB_CLIENT_ID'),
    client_secret=config.GITHUB_CLIENT_SECRET or os.getenv('GITHUB_CLIENT_SECRET'),
    authorize_url='https://github.com/login/oauth/authorize',
    access_token_url='https://github.com/login/oauth/access_token',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},  # Minimal default - frontend will override with specific scopes
)

