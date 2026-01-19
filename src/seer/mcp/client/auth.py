"""OAuth 2.1 PKCE flow handler for MCP clients."""

import base64
import hashlib
import secrets
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from seer.mcp.client.token_store import TokenStore


class OAuthError(Exception):
    """OAuth authentication error."""


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    authorization_code: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self):  # pylint: disable=invalid-name  # Reason: Required by BaseHTTPRequestHandler
        """Handle GET request with authorization code."""
        # Parse query parameters
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        # Extract authorization code or error
        if "code" in params:
            OAuthCallbackHandler.authorization_code = params["code"][0]
            OAuthCallbackHandler.state = params.get("state", [None])[0]

            # Send success response
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <head><title>Authorization Successful</title></head>
                <body style="font-family: system-ui; text-align: center; padding: 50px;">
                    <h1>Authorization Successful!</h1>
                    <p>You can close this window and return to your IDE.</p>
                    <script>setTimeout(() => window.close(), 2000);</script>
                </body>
                </html>
            """)
        elif "error" in params:
            OAuthCallbackHandler.error = params["error"][0]
            error_description = params.get("error_description", ["Unknown error"])[0]

            # Send error response
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"""
                <html>
                <head><title>Authorization Failed</title></head>
                <body style="font-family: system-ui; text-align: center; padding: 50px;">
                    <h1>Authorization Failed</h1>
                    <p>{error_description}</p>
                    <p>You can close this window and try again.</p>
                </body>
                </html>
            """.encode())
        else:
            # Invalid callback
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"Invalid callback")

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin  # Reason: Required by BaseHTTPRequestHandler
        """Suppress HTTP server logs."""


class OAuthHandler:
    """Handles OAuth 2.1 PKCE flow for MCP clients."""

    def __init__(self, api_url: str, token_store: TokenStore, client_id: str = "seer-mcp-client"):
        """Initialize OAuth handler.

        Args:
            api_url: Seer API base URL
            token_store: Token storage instance
            client_id: OAuth client identifier
        """
        self.api_url = api_url.rstrip("/")
        self.token_store = token_store
        self.client_id = client_id

    def _generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code_verifier and code_challenge.

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        # Generate random code_verifier (43-128 chars)
        code_verifier = secrets.token_urlsafe(64)

        # Generate code_challenge = base64url(SHA256(code_verifier))
        digest = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")

        return code_verifier, code_challenge

    def _get_free_port(self) -> int:
        """Find a free port for the callback server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def start_auth_flow(self, scope: str = "workflow:read workflow:write workflow:execute integration:read integration:write") -> None:
        """Start OAuth 2.1 authorization flow with PKCE.

        This method:
        1. Generates PKCE code_verifier and code_challenge
        2. Starts a local HTTP server for the callback
        3. Opens the browser to the authorization page
        4. Waits for the callback with authorization code
        5. Exchanges code for tokens
        6. Saves tokens securely

        Args:
            scope: Requested OAuth scopes (space-separated)

        Raises:
            Exception: If authorization fails
        """
        # Generate PKCE pair
        code_verifier, code_challenge = self._generate_pkce_pair()

        # Start local callback server
        port = self._get_free_port()
        redirect_uri = f"http://localhost:{port}/callback"

        # Build authorization URL
        auth_params = {
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "scope": scope,
            "state": secrets.token_urlsafe(16),
        }
        auth_url = f"{self.api_url}/api/oauth/mcp/authorize?{urlencode(auth_params)}"

        # Reset handler state
        OAuthCallbackHandler.authorization_code = None
        OAuthCallbackHandler.state = None
        OAuthCallbackHandler.error = None

        # Open browser
        print(f"Opening browser for authorization: {auth_url}")
        webbrowser.open(auth_url)

        # Start callback server
        server = HTTPServer(("localhost", port), OAuthCallbackHandler)
        server.timeout = 300  # 5 minute timeout

        # Wait for callback
        print(f"Waiting for authorization callback on http://localhost:{port}/callback...")
        server.handle_request()
        server.server_close()

        # Check for errors
        if OAuthCallbackHandler.error:
            raise OAuthError(f"Authorization failed: {OAuthCallbackHandler.error}")

        if not OAuthCallbackHandler.authorization_code:
            raise OAuthError("No authorization code received")

        authorization_code = OAuthCallbackHandler.authorization_code

        # Exchange code for tokens
        print("Exchanging authorization code for tokens...")
        token_response = self._exchange_code_for_tokens(
            authorization_code,
            code_verifier,
            redirect_uri,
        )

        # Save tokens
        self.token_store.save_tokens(
            access_token=token_response["access_token"],
            refresh_token=token_response["refresh_token"],
            expires_in=token_response["expires_in"],
            api_url=self.api_url,
            scope=token_response["scope"],
        )

        print("Authorization successful! Tokens saved securely.")

    def _exchange_code_for_tokens(
        self,
        code: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> dict:
        """Exchange authorization code for access and refresh tokens.

        Args:
            code: Authorization code from callback
            code_verifier: PKCE code verifier
            redirect_uri: Redirect URI used in authorization

        Returns:
            Token response dict with access_token, refresh_token, expires_in, scope

        Raises:
            Exception: If token exchange fails
        """
        token_url = f"{self.api_url}/api/oauth/mcp/token"

        payload = {
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
        }

        with httpx.Client() as client:
            response = client.post(token_url, json=payload, timeout=30)

            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise OAuthError(f"Token exchange failed: {error_detail}")

            return response.json()

    def refresh_tokens(self, refresh_token: str) -> dict:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token from previous token response

        Returns:
            Token response dict with new access_token

        Raises:
            Exception: If refresh fails
        """
        refresh_url = f"{self.api_url}/api/oauth/mcp/refresh"

        payload = {
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }

        with httpx.Client() as client:
            response = client.post(refresh_url, json=payload, timeout=30)

            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise OAuthError(f"Token refresh failed: {error_detail}")

            return response.json()

    def revoke_token(self, token: str) -> None:
        """Revoke an access or refresh token.

        Args:
            token: Token to revoke

        Raises:
            Exception: If revocation fails
        """
        revoke_url = f"{self.api_url}/api/oauth/mcp/revoke"

        payload = {
            "token": token,
            "client_id": self.client_id,
        }

        with httpx.Client() as client:
            response = client.post(revoke_url, json=payload, timeout=30)

            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise OAuthError(f"Token revocation failed: {error_detail}")
