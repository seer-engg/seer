"""
Seer MCP Server - FastMCP server with stdio and HTTP transport support.

Usage:
    # stdio transport (for Claude Code, Claude Desktop)
    seer-mcp --transport stdio

    # SSE transport (for MCP Inspector testing)
    seer-mcp --transport sse --port 9001
    # Then connect: npx @modelcontextprotocol/inspector --transport sse --server-url http://localhost:9001/sse

    # HTTP transport (for ChatGPT/OpenAI Apps)
    seer-mcp --transport http --port 9001
    # Endpoint: http://localhost:9001/mcp
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)

# Create the FastMCP server instance
mcp = FastMCP(
    name="Seer Workflow Server",
    instructions="Seer MCP Server - Manage workflows, discover tools/triggers, and execute automation.",
)


async def oauth_protected_resource_metadata(request):
    """
    Serve OAuth Protected Resource Metadata for ChatGPT/MCP client discovery.

    This endpoint tells MCP clients where to authenticate.
    See: https://datatracker.ietf.org/doc/html/rfc9728
    """
    # Get the server URL from config or construct from request
    resource_url = config.mcp_server_url
    scheme = config.redirect_uri_scheme
    if not resource_url:
        # Fallback: construct from request
        resource_url = f"{scheme}://{request.url.netloc}"

    # pylint: disable=no-member # Reason: Pydantic resolves Optional[str] at runtime, not FieldInfo
    scopes = [s.strip() for s in config.mcp_oauth_scopes.split(",") if s.strip()]

    metadata = {
        "resource": resource_url,
        "authorization_servers": [config.mcp_oauth_authorization_server],
        "scopes_supported": scopes,
    }

    # Add documentation URL if configured
    if config.mcp_resource_documentation:
        metadata["resource_documentation"] = config.mcp_resource_documentation

    return JSONResponse(metadata)


_MCP_TOOLS_REGISTERED = False


def _register_tools() -> None:
    """Register all MCP tools with the server. Idempotent — safe to call multiple times."""
    global _MCP_TOOLS_REGISTERED  # pylint: disable=global-statement # Reason: Idempotent guard for module-level mcp singleton
    if _MCP_TOOLS_REGISTERED:
        return
    _MCP_TOOLS_REGISTERED = True

    # pylint: disable=import-outside-toplevel,unused-import # Reason: Lazy loading to avoid circular imports

    # Unified tools (discovery + templates) — registered via factory pattern
    from seer.tools.unified_tools import register_unified_tools
    register_unified_tools()
    from seer.tools.tool_factory import unified_registry
    unified_registry.register_mcp_tools(mcp)

    # MCP-only modules (not yet in factory)


def _create_auth_middleware():
    """
    Create auth middleware for MCP using opaque token validation.

    MCP clients (like ChatGPT) receive Clerk OAuth opaque tokens that must
    be validated via the /oauth/userinfo endpoint, not via local JWT verification.

    Returns:
        Middleware instance or None if OAuth authorization server is not configured
    """
    if not config.mcp_oauth_authorization_server:
        logger.info("MCP server running without authentication (mcp_oauth_authorization_server not configured)")
        return None

    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.auth.clerk_verifier import ClerkOpaqueTokenVerifier
    from seer.mcp.auth import MCPOpaqueAuthMiddleware

    # Construct userinfo URL from the authorization server
    userinfo_url = f"{config.mcp_oauth_authorization_server}/oauth/userinfo"

    verifier = ClerkOpaqueTokenVerifier(
        userinfo_url=userinfo_url,
        timeout=10.0,
    )
    logger.info("MCP server using Clerk opaque token authentication via %s", userinfo_url)
    return Middleware(MCPOpaqueAuthMiddleware, verifier=verifier)


def create_combined_mcp_app():
    """
    Create a combined MCP Starlette app for mounting into FastAPI.

    This creates a single app that handles both SSE (/sse) and HTTP (/mcp) transports.
    Includes MCP-specific auth middleware if Clerk is configured.

    Returns:
        tuple: (app, lifespan_cm) where app is the Starlette app and lifespan_cm is the
               combined lifespan context manager to be nested in the parent app's lifespan
    """
    _register_tools()

    # Get MCP apps for both transports
    # These apps have internal routes at /sse and /mcp respectively
    mcp_sse_app = mcp.http_app(transport="sse")  # type: ignore[arg-type]
    mcp_http_app = mcp.http_app(transport="http")  # type: ignore[arg-type]

    # Combine routes from both apps into a single Starlette app
    combined_routes = list(mcp_sse_app.routes) + list(mcp_http_app.routes)

    # Create combined lifespan that runs both app lifespans
    @asynccontextmanager
    async def combined_lifespan(app):
        async with mcp_sse_app.lifespan(app):
            async with mcp_http_app.lifespan(app):
                yield

    # Build middleware list - MCP auth if Clerk is configured
    middleware_list = []
    auth_middleware = _create_auth_middleware()
    if auth_middleware:
        middleware_list.append(auth_middleware)
        logger.info("MCP combined app using Clerk JWT authentication")
    else:
        logger.info("MCP combined app running without authentication")

    starlette_app = Starlette(
        routes=combined_routes,
        lifespan=combined_lifespan,
        middleware=middleware_list,
    )

    # Return both the app and the lifespan for the parent to use
    return starlette_app, combined_lifespan


def create_http_app(transport: str = "sse") -> Starlette:
    """
    Create a Starlette app that combines:
    - OAuth Protected Resource Metadata endpoint (/.well-known/oauth-protected-resource)
    - FastMCP HTTP/SSE routes (mounted from mcp.http_app())
    - Optional JWT authentication middleware (when Clerk is configured)

    Args:
        transport: MCP transport type - "sse" (default, for MCP Inspector) or "http" (streamable HTTP)
    """
    # Get the FastMCP HTTP app with specified transport
    # SSE transport: exposes /sse endpoint (compatible with MCP Inspector)
    # HTTP transport: exposes /mcp endpoint (streamable HTTP for ChatGPT)
    mcp_http_app = mcp.http_app(transport=transport)  # type: ignore[arg-type]

    # Define routes - OAuth discovery + MCP routes mounted at root
    # IMPORTANT: Mount must be included in routes list, and lifespan must be passed
    # for FastMCP session manager to initialize properly
    routes = [
        Route(
            "/.well-known/oauth-protected-resource",
            oauth_protected_resource_metadata,
            methods=["GET"]
        ),
        # Mount MCP HTTP app at root - handles /sse or /mcp depending on transport
        Mount("", app=mcp_http_app),
    ]

    # Build middleware list
    middleware_list = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    # Add auth middleware if Clerk is configured
    # temporarily disabled until we can verify it works with both SSE and HTTP transports without interfering with streaming responses
    auth_middleware = _create_auth_middleware()
    if auth_middleware:
        middleware_list.append(auth_middleware)

    # Create the combined Starlette app
    # CRITICAL: Pass lifespan from FastMCP app - without this, session manager won't initialize
    app = Starlette(
        routes=routes,
        middleware=middleware_list,
        lifespan=mcp_http_app.lifespan,
    )

    return app


def main() -> None:
    """CLI entry point for seer-mcp command."""
    # Initialize Sentry for standalone MCP server mode
    if config.is_sentry_configured:
        from seer.observability.sentry_client import init_sentry  # pylint: disable=import-outside-toplevel  # Reason: lazy import for optional dependency
        if init_sentry():
            logger.info("Sentry error monitoring initialized for MCP server")

    parser = argparse.ArgumentParser(
        description="Seer MCP Server - Workflow management via Model Context Protocol"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport mode: stdio (default), sse (for MCP Inspector, /sse endpoint), http (for ChatGPT, /mcp endpoint)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9001,
        help="Port for HTTP/SSE transport (default: 9001)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for HTTP/SSE transport (default: 127.0.0.1)"
    )

    args = parser.parse_args()

    # Register all tools
    _register_tools()

    if args.transport == "stdio":
        logger.info("Starting MCP server with stdio transport")
        mcp.run(transport="stdio")
    elif args.transport in ("http", "sse"):
        # Map CLI transport arg to FastMCP transport type
        fastmcp_transport = "sse" if args.transport == "sse" else "http"
        endpoint_path = "/sse" if fastmcp_transport == "sse" else "/mcp"

        logger.info("Starting MCP server with %s transport on %s:%d", fastmcp_transport.upper(), args.host, args.port)
        logger.info("MCP endpoint: http://%s:%d%s", args.host, args.port, endpoint_path)
        logger.info("OAuth discovery: http://%s:%d/.well-known/oauth-protected-resource", args.host, args.port)

        # Create combined app with OAuth discovery + MCP transport
        app = create_http_app(transport=fastmcp_transport)

        # Run with uvicorn
        import uvicorn  # pylint: disable=import-outside-toplevel # Reason: Only needed for HTTP transport
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        print(f"Unknown transport: {args.transport}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
