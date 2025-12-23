from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import RedirectResponse
from .oauth import oauth
from .services import store_oauth_connection, list_connections, disconnect_provider, delete_connection_by_id
import json
import base64
import os
import logging
from shared.logger import get_logger
from shared.database.models import User
logger = get_logger("api.integrations.router")

router = APIRouter(prefix="/integrations", tags=["integrations"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

def encode_state(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

def decode_state(state: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(state).decode())

@router.get("/{provider}/connect")
async def connect(
    request: Request,
    provider: str,
    redirect_to: str = Query(None),
    scope: str = Query(...),  # OAuth scope from frontend (REQUIRED - frontend controls scopes)
):
    """
    Start OAuth flow for a provider.
    
    Args:
        provider: Provider name (google, github, googledrive, gmail)
        redirect_to: Redirect URL after auth
        scope: OAuth scope from frontend (REQUIRED - frontend controls which scopes to request)
    
    Note:
        Frontend must always pass scope parameter. This ensures frontend controls
        which permissions are requested (read-only is core differentiation).
    """
    if provider not in ['google', 'github', 'googledrive', 'gmail']:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    
    if not scope:
        raise HTTPException(status_code=400, detail="scope parameter is required. Frontend must specify OAuth scopes.")
    
    real_provider = 'google' if provider in ['googledrive', 'gmail'] else provider
    
    redirect_uri = request.url_for('auth_callback', provider=provider)
    
    # Store user_id, scope, and final redirect in state
    # Scope is stored so we can save it when token is received
    user:User = request.state.db_user
    state_data = {
        'user_id': user.user_id,
        'user_email': user.email,
        'redirect_to': redirect_to or f"{FRONTEND_URL}/settings/integrations",
        'original_provider': provider,
        'requested_scope': scope  # Store requested scope to save in callback
    }

    logger.info(f"State data: {state_data}")
    state = encode_state(state_data)
    
    client = oauth.create_client(real_provider)
    # Always pass scope from frontend - no defaults
    kwargs = {'state': state, 'scope': scope}
        
    return await client.authorize_redirect(request, redirect_uri, **kwargs)

@router.get("/{provider}/callback", name="auth_callback")
async def auth_callback(request: Request, provider: str):
    # provider param here comes from the redirect_uri path.
    # If we started with 'googledrive', redirect_uri was /integrations/googledrive/callback
    # So 'provider' is 'googledrive'.
    
    real_provider = 'google' if provider in ['googledrive', 'gmail'] else provider
    client = oauth.create_client(real_provider)
    try:
        token = await client.authorize_access_token(request)
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    # Retrieve user_id from state
    # Authlib validates state match, but we need to extract data from it.
    # The 'state' query param is in the request.
    state = request.query_params.get('state')
    if not state:
         raise HTTPException(status_code=400, detail="Missing state")
         
    try:
        state_data = decode_state(state)
        user_id = state_data.get('user_id')
        redirect_to = state_data.get('redirect_to')
        requested_scope = state_data.get('requested_scope')  # Get requested scope from state
    except:
        raise HTTPException(status_code=400, detail="Invalid state")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id in state")
        
    # Get user profile
    if provider == 'google' or provider in ['googledrive', 'gmail']:
        # Try to parse id_token if present, otherwise fetch from userinfo endpoint
        logger.info(f"Token is to parse: {token}")
        # if 'id_token' in token:
        #     logger.info(f"Parsing id_token")
        #     user_info = await client.parse_id_token(request, token)
        #     logger.info(f"User info parsed: {user_info}")
        # else:
            # Fallback to userinfo endpoint when id_token is not returned
        resp = await client.get('https://www.googleapis.com/oauth2/v3/userinfo', token=token)
        user_info = resp.json()
    elif provider == 'github':
        resp = await client.get('user', token=token)
        user_info = resp.json()
    else:
        user_info = {}
    
    # Extract granted scopes from token response
    # Token may contain 'scope' field (space-separated) or we use requested_scope
    granted_scopes = token.get('scope') or requested_scope or ''
    
    await store_oauth_connection(user_id, provider, token, user_info, granted_scopes)
    
    return RedirectResponse(url=f"{redirect_to}?connected={provider}")

@router.post("/{provider}/disconnect")
async def disconnect(provider: str, request: Request):
    user:User = request.state.db_user
    await disconnect_provider(user, provider)
    return {"status": "success"}

@router.delete("/{connection_id}")
async def delete_connection(connection_id: str, request: Request):
    user:User = request.state.db_user
    await delete_connection_by_id(user, connection_id)
    return {"status": "success"}

@router.get("/")
async def list_integrations(request: Request):
    user:User = request.state.db_user
    logger.info(f"Listing integrations for user {user.user_id}")
    connections = await list_connections(user)
    res = []
    for conn in connections:
        # Construct composite ID so frontend can use it for deletion if needed
        # Or just use DB ID if we handle it in delete_connection_by_id
        composite_id = f"{conn.provider}:{conn.id}"
        
        # Determine toolkit/provider logic for frontend compat
        # Frontend expects 'toolkit': {'slug': ...}
        
        res.append({
            "id": composite_id, 
            "status": "ACTIVE" if conn.status == 'active' else "INACTIVE",
            "user_id": user.user_id,
            "toolkit": {
                "slug": conn.provider
            },
            "connection": {
                "user_id": user.user_id,
                "provider_account_id": conn.provider_account_id
            }
        })
    return {"items": res}

