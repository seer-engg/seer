from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import RedirectResponse
from .oauth import oauth
from .services import store_oauth_connection, list_connections, disconnect_provider, delete_connection_by_id
import json
import base64
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integrations", tags=["integrations"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

def encode_state(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

def decode_state(state: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(state).decode())

@router.get("/{provider}/connect")
async def connect(request: Request, provider: str, user_id: str = Query(...), redirect_to: str = Query(None)):
    """
    Start OAuth flow for a provider.
    """
    if provider not in ['google', 'github', 'googledrive', 'gmail']:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    
    real_provider = 'google' if provider in ['googledrive', 'gmail'] else provider
    
    redirect_uri = request.url_for('auth_callback', provider=provider)
    
    # Scopes
    scope = None
    if provider == 'googledrive':
        scope = 'openid email profile https://www.googleapis.com/auth/drive.readonly'
    elif provider == 'gmail':
        scope = 'openid email profile https://www.googleapis.com/auth/gmail.readonly'
    
    # Store user_id and final redirect in state
    state_data = {
        'user_id': user_id,
        'redirect_to': redirect_to or f"{FRONTEND_URL}/settings/integrations",
        'original_provider': provider
    }
    
    state = encode_state(state_data)
    
    client = oauth.create_client(real_provider)
    kwargs = {'state': state}
    if scope:
        kwargs['scope'] = scope
        
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
    except:
        raise HTTPException(status_code=400, detail="Invalid state")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id in state")
        
    # Get user profile
    if provider == 'google':
        user_info = await client.parse_id_token(request, token)
        # Or fetch userinfo endpoint if needed
        # resp = await client.get('https://www.googleapis.com/oauth2/v3/userinfo', token=token)
        # user_info = resp.json()
        # id_token usually has enough info
    elif provider == 'github':
        resp = await client.get('user', token=token)
        user_info = resp.json()
    else:
        user_info = {}
        
    await store_oauth_connection(user_id, provider, token, user_info)
    
    return RedirectResponse(url=f"{redirect_to}?connected={provider}")

@router.post("/{provider}/disconnect")
async def disconnect(provider: str, user_id: str = Query(...)):
    await disconnect_provider(user_id, provider)
    return {"status": "success"}

@router.delete("/{connection_id}")
async def delete_connection(connection_id: str, user_id: str = Query(...)):
    await delete_connection_by_id(user_id, connection_id)
    return {"status": "success"}

@router.get("/")
async def list_integrations(user_id: str = Query(...)):
    connections = await list_connections(user_id)
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
            "user_id": user_id,
            "toolkit": {
                "slug": conn.provider
            },
            "connection": {
                "user_id": user_id,
                "provider_account_id": conn.provider_account_id
            }
        })
    return {"items": res}

