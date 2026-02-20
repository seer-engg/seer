# pylint: disable=duplicate-code  # Reason: SettingsConfigDict block shared with observability.constants for consistent env loading
"""
Type-safe configuration for Seer using Pydantic Settings.

This module provides a centralized, type-safe configuration system with the following priority:
1. Environment variables
2. .env file
3. AWS Parameter Store (if configured)
4. Default values

This ensures local development works with defaults while production can use AWS Parameter Store.

Usage:
    from seer.config import config

    if score >= config.eval_pass_threshold:
        ...
"""
from typing import Optional, Tuple, Type

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from seer.config_properties import SeerConfigPropertiesMixin
from seer.utilities.aws.parameter_store import AwsSsmSettingsSource


class SeerConfig(SeerConfigPropertiesMixin, BaseSettings):
    """
    Central configuration for Seer.

    All configuration is loaded from environment variables or .env file.
    Provides type safety and validation at startup.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = Field(
        default="dev", description="Environment"
    )

    #### SECRETS ####

    # ============================================================================
    # API Keys & Authentication
    # ============================================================================

    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key for LLM and embeddings"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key for Claude models"
    )
    openrouter_api_key: Optional[str] = Field(
        default=None, description="OpenRouter API key for multi-provider LLM access"
    )
    tavily_api_key: Optional[str] = Field(
        default=None, description="Tavily API key for web search"
    )

    # ============================================================================
    # LangGraph Checkpointer Configuration
    # ============================================================================

    database_url: Optional[str] = Field(
        default=None,
        description=(
            "PostgreSQL connection string for LangGraph checkpointer "
            "(e.g., postgresql://user:pass@host:port/db). "
            "Required for human-in-the-loop interrupts."
        )
    )
    db_max_connections: int = Field(
        default=10, description="Maximum number of database connections"
    )
    db_min_connections: int = Field(
        default=1, description="Minimum number of database connections"
    )
    db_generate_schemas: bool = Field(
        default=False, description="Generate database schemas"
    )


    # ============================================================================
    # Clerk Authentication Configuration
    # ============================================================================

    clerk_jwks_url: Optional[str] = Field(
        default=None, description="Clerk JWKS URL for JWT verification"
    )
    clerk_issuer: Optional[str] = Field(
        default=None, description="Clerk JWT issuer (e.g., https://clerk.your-domain.com)"
    )
    clerk_audience: Optional[str] = Field(
        default=None, description="Clerk JWT audience (e.g., ['api.your-domain.com'])"
    )



    ### Flags ####

    # ============================================================================
    # PostgreSQL Tool Autonomy Configuration
    # ============================================================================

    postgres_write_requires_approval: bool = Field(
        default=True,
        description=(
            "If True, PostgreSQL write operations (INSERT, UPDATE, DELETE, DDL) "
            "require human approval via interrupt before execution. "
            "Read operations are always allowed."
        )
    )

    # Vector embeddings configuration
    embedding_dims: int = Field(
        default=1536, description="OpenAI embedding dimensions"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    embedding_batch_size: int = Field(
        default=128, description="OpenAI embedding batch size"
    )

    # ============================================================================
    # Deployment Mode Configuration
    # ============================================================================

    seer_mode: str = Field(
        default="self-hosted", description="Deployment mode: 'self-hosted' or 'cloud'"
    )



    default_llm_model: str = Field(default="moonshotai/kimi-k2.5", description="Default LLM model")

    # Taskiq / Valkey configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Valkey/Redis connection string for Taskiq broker and result backend. Use 'rediss://' for TLS/SSL connections.",
    )
    redis_socket_timeout: float = Field(
        default=30.0,
        description="Socket timeout for Redis operations in seconds.",
    )
    redis_socket_connect_timeout: float = Field(
        default=5.0,
        description="Connection timeout for Redis in seconds.",
    )
    redis_health_check_interval: int = Field(
        default=30,
        description="Interval for Redis connection health checks in seconds (0 to disable).",
    )
    redis_max_connections: int = Field(
        default=20,
        description="Maximum Redis connection pool size.",
    )
    redis_socket_keepalive: bool = Field(
        default=True,
        description="Enable TCP keepalive for Redis connections to prevent idle timeout disconnections.",
    )

    google_client_id: str = Field(default="", description="Google OAuth client ID")
    google_client_secret: str = Field(default="", description="Google OAuth client secret")

    github_client_id: Optional[str] = Field(
        default=None, description="GitHub OAuth client ID"
    )
    github_client_secret: Optional[str] = Field(
        default=None, description="GitHub OAuth client secret"
    )

    supabase_client_id: Optional[str] = Field(
        default=None, description="Supabase management OAuth client ID"
    )
    supabase_client_secret: Optional[str] = Field(
        default=None, description="Supabase management OAuth client secret"
    )
    supabase_management_api_base: str = Field(
        default="https://api.supabase.com",
        description="Supabase management API base URL",
    )

    discord_client_id: Optional[str] = Field(
        default=None, description="Discord OAuth client ID"
    )
    discord_client_secret: Optional[str] = Field(
        default=None, description="Discord OAuth client secret"
    )
    discord_bot_token: Optional[str] = Field(
        default=None, description="Discord bot token for API calls"
    )

    linkedin_client_id: Optional[str] = Field(
        default=None, description="LinkedIn OAuth client ID"
    )
    linkedin_client_secret: Optional[str] = Field(
        default=None, description="LinkedIn OAuth client secret"
    )

    slack_client_id: Optional[str] = Field(
        default=None, description="Slack OAuth client ID"
    )
    slack_client_secret: Optional[str] = Field(
        default=None, description="Slack OAuth client secret"
    )

    # ============================================================================
    # Feature Flags
    # ============================================================================

    auto_open_browser: bool = Field(
        default=True,
        description="Automatically open frontend in browser on server startup (self-hosted mode only)"
    )

    nexus_max_agent_steps: int = Field(
        default=75,
        description="Default maximum agent steps for Nexus chat (LangGraph recursion_limit)"
    )

    # ============================================================================
    # Trigger Poller
    # ============================================================================
    trigger_poller_enabled: bool = Field(
        default=True,
        description="Enable background polling for provider-based workflow triggers.",
    )
    trigger_poller_interval_seconds: int = Field(
        default=5,
        description="Sleep interval between poll engine ticks.",
    )
    trigger_poller_max_batch_size: int = Field(
        default=10,
        description="Maximum subscriptions to lease per poll tick.",
    )
    trigger_poller_lock_timeout_seconds: int = Field(
        default=60,
        description="Lease timeout for poll locks in seconds.",
    )

    # ============================================================================
    # Workflow File System (S3/R2)
    # ============================================================================
    # NOTE: AWS credentials use standard boto3 credential chain:
    # - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or IAM roles)
    # - AWS_REGION / AWS_DEFAULT_REGION (defaults to us-east-1)
    workflow_file_s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 bucket name for workflow file storage",
    )
    workflow_file_s3_endpoint_url: Optional[str] = Field(
        default=None,
        description="Custom endpoint URL for S3-compatible storage (Cloudflare R2, MinIO, etc.)",
    )
    workflow_file_max_size_mb: int = Field(
        default=100,
        description="Maximum file size in MB for workflow files",
    )
    workflow_file_presigned_url_expiry_seconds: int = Field(
        default=3600,
        description="Default expiry time for presigned URLs in seconds",
    )

    # ============================================================================
    # Request Profiling
    # ============================================================================
    request_profiling_enabled: bool = Field(
        default=False,
        description="Enable pyinstrument profiling for each API request (development only)."
    )
    request_profiling_output_dir: str = Field(
        default="./data/profiles/noposthog",
        description="Directory path to write pyinstrument HTML reports."
    )

    webhook_base_url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL for webhook callbacks (e.g., https://seer.example.com). "
            "Used by external services to send webhooks. "
            "Defaults to http://localhost:8000 if not set."
        ),
    )
    redirect_uri_scheme: str = Field(
        default="http",
        description="Scheme for redirect URIs (e.g., https or http)"
    )

    # ============================================================================
    # Stripe Subscription Configuration
    # ============================================================================

    stripe_secret_key: Optional[str] = Field(
        default=None,
        description="Stripe secret API key (sk_test_... or sk_live_...)"
    )
    stripe_publishable_key: Optional[str] = Field(
        default=None,
        description="Stripe publishable key for frontend (pk_test_... or pk_live_...)"
    )
    stripe_webhook_secret: Optional[str] = Field(
        default=None,
        description="Stripe webhook signing secret (whsec_...)"
    )
    clerk_secret_key: Optional[str] = Field(
        default=None,
        description="Clerk secret key for updating user metadata"
    )
    frontend_url: str = Field(
        default="https://app.getseer.dev",
        description="Frontend application URL (e.g., OAuth redirects, checkout redirects)"
    )

    # ============================================================================
    # Slack Error Notifications Configuration
    # ============================================================================

    slack_bot_token: Optional[str] = Field(
        default=None,
        description="Slack Bot OAuth token (xoxb-...) for sending error notifications"
    )
    slack_error_channel_id: Optional[str] = Field(
        default=None,
        description="Slack channel ID (C01234567890) for error notifications"
    )
    slack_notifications_enabled: bool = Field(
        default=False,
        description="Enable Slack notifications for ERROR and CRITICAL log levels"
    )

    # ===========================================================================
    # MLflow Configuration
    # ============================================================================
    mlflow_enabled: bool = Field(
        default=False, description="Enable MLflow autologging for LangChain"
    )

    # ============================================================================
    # MCP Server OAuth Configuration (for ChatGPT integration)
    # ============================================================================

    mcp_server_url: Optional[str] = Field(
        default=None,
        description="Base URL of the MCP server (e.g., https://mcp.getseer.dev)"
    )
    mcp_oauth_authorization_server: str = Field(
        default="https://nice-ram-42.clerk.accounts.dev",
        description="Clerk authorization server URL for MCP OAuth"
    )
    mcp_oauth_scopes: str = Field(
        default="profile,email",
        description="Comma-separated list of OAuth scopes supported by MCP server"
    )
    mcp_resource_documentation: Optional[str] = Field(
        default=None,
        description="URL to MCP server documentation"
    )
    mcp_enabled: bool = Field(
        default=True,
        description="Enable MCP endpoints (/sse, /mcp) in the main API server"
    )

    # ============================================================================
    # Browser Pool Configuration
    # ============================================================================

    browser_pool_max_concurrent: int = Field(
        default=5,
        description="Maximum number of concurrent browser sessions in the pool"
    )
    browser_pool_default_timeout_seconds: int = Field(
        default=300,
        description="Default timeout in seconds for browser pool sessions"
    )
    browser_pool_reaper_interval_seconds: int = Field(
        default=30,
        description="Interval in seconds for the session reaper to check for expired sessions"
    )
    browser_session_encryption_key: Optional[str] = Field(
        default=None,
        description="Fernet encryption key for browser session state. If not set, derived from SECRET_KEY."
    )

    # Browser Live Streaming
    browser_screencast_quality: int = Field(
        default=60, description="JPEG quality for CDP screencast (1-100)"
    )
    browser_screencast_max_width: int = Field(
        default=1280, description="Max screencast frame width"
    )
    browser_screencast_max_height: int = Field(
        default=800, description="Max screencast frame height"
    )
    browser_screencast_every_nth_frame: int = Field(
        default=1, description="Send every Nth frame (1=all)"
    )
    browser_interactive_timeout_seconds: int = Field(
        default=600, description="Interactive session timeout"
    )

    # Session Recording
    browser_recording_enabled: bool = Field(
        default=True, description="Enable rrweb recording"
    )
    browser_recording_max_events: int = Field(
        default=50000, description="Max rrweb events per recording"
    )
    browser_recording_max_size_mb: int = Field(
        default=50, description="Max compressed recording size MB"
    )
    browser_recording_rrweb_cdn_url: str = Field(
        default="https://cdn.jsdelivr.net/npm/rrweb@2.0.0-alpha.13/dist/record/rrweb-record.min.js",
        description="CDN URL for rrweb recording script",
    )

    # Browser Stealth
    browser_stealth_enabled: bool = Field(
        default=True,
        description="Enable stealth mode (--headless=new) for interactive browser sessions",
    )

    # ============================================================================
    # Langfuse Configuration
    # ============================================================================
    langfuse_enabled: bool = Field(
        default=False, description="Enable Langfuse tracing for LangChain/LangGraph"
    )
    langfuse_host: Optional[str] = Field(
        default=None, description="Langfuse host URL (e.g., https://cloud.langfuse.com)"
    )
    # Nexus Agent tracing (separate project)
    langfuse_nexus_public_key: Optional[str] = Field(
        default=None, description="Langfuse public key for Nexus agent project (pk-lf-...)"
    )
    langfuse_nexus_secret_key: Optional[str] = Field(
        default=None, description="Langfuse secret key for Nexus agent project (sk-lf-...)"
    )
    # Workflow/Compiler tracing (separate project)
    langfuse_workflow_public_key: Optional[str] = Field(
        default=None, description="Langfuse public key for Workflow project (pk-lf-...)"
    )
    langfuse_workflow_secret_key: Optional[str] = Field(
        default=None, description="Langfuse secret key for Workflow project (sk-lf-...)"
    )

    # ============================================================================
    # PostHog Analytics Configuration
    # ============================================================================

    posthog_api_key: Optional[str] = Field(
        default=None, description="PostHog API key for analytics"
    )
    posthog_host: str = Field(
        default="https://us.i.posthog.com", description="PostHog instance host URL"
    )
    posthog_enabled: bool = Field(
        default=False, description="Enable/disable PostHog analytics"
    )

    # ============================================================================
    # Sentry Error Monitoring Configuration
    # ============================================================================

    sentry_dsn: Optional[str] = Field(
        default=None, description="Sentry DSN for error monitoring"
    )
    sentry_environment: Optional[str] = Field(
        default=None, description="Sentry environment tag (defaults to env if not set)"
    )
    sentry_traces_sample_rate: float = Field(
        default=0.1, description="Sentry performance trace sampling rate (0.0-1.0)"
    )
    sentry_profiles_sample_rate: float = Field(
        default=0.1, description="Sentry profile sampling rate (0.0-1.0)"
    )
    sentry_enabled: bool = Field(
        default=True, description="Enable Sentry (requires DSN to be set)"
    )

    # ============================================================================
    # Memory Layer (Mem0) Configuration
    # ============================================================================

    memory_enabled: bool = Field(
        default=True,
        description="Enable Mem0 memory layer for cross-session user context"
    )
    mem0_vector_store: str = Field(
        default="pgvector",
        description="Vector store provider for Mem0: pgvector (recommended), qdrant, chroma, etc."
    )
    mem0_qdrant_host: str = Field(
        default="localhost",
        description="Qdrant host for Mem0 vector store"
    )
    mem0_qdrant_port: int = Field(
        default=6333,
        description="Qdrant port for Mem0 vector store"
    )
    mem0_collection_name: str = Field(
        default="nexus_user_memories",
        description="Collection name for Mem0 vector store"
    )
    memory_context_injection_enabled: bool = Field(
        default=True,
        description="Auto-inject relevant memories into agent system prompt"
    )
    memory_context_max_memories: int = Field(
        default=10,
        description="Maximum number of memories to inject into system prompt"
    )
    memory_extraction_enabled: bool = Field(
        default=True,
        description="Auto-extract memories from completed chat sessions"
    )
    memory_extraction_model: str = Field(
        default="moonshotai/kimi-k2.5",
        description="LLM model for memory extraction (used by Mem0, works with OpenRouter)"
    )
    mem0_llm_provider: str = Field(
        default="openrouter",
        description="LLM provider for Mem0: 'openai', 'openrouter', or 'ollama'"
    )
    mem0_llm_base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for Mem0 LLM (auto-set for openrouter)"
    )
    mem0_embedder_provider: str = Field(
        default="huggingface",
        description="Embedder provider for Mem0: 'openai', 'huggingface', or 'ollama'"
    )
    mem0_embedder_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for Mem0 (default: free HuggingFace model)"
    )

    @classmethod
    def settings_customise_sources(  # pylint: disable=too-many-positional-arguments  # Reason: Method signature is defined by Pydantic's BaseSettings API and cannot be modified
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize the priority order of settings sources.

        Priority (highest to lowest):
        1. init_settings: Values passed to __init__
        2. env_settings: Environment variables
        3. dotenv_settings: .env file
        4. AWS Parameter Store: Fallback for production secrets
        5. Pydantic defaults: Default values defined in Field()

        This ensures:
        - Local development can use .env files or defaults
        - Production can use AWS Parameter Store without .env files
        - AWS credentials are optional (graceful fallback for local dev)
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            AwsSsmSettingsSource(settings_cls),  # AWS Parameter Store fallback
            file_secret_settings,
        )


# ============================================================================
# Global Config Instance
# ============================================================================

config = SeerConfig()
