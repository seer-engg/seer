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

from seer.utilities.aws.parameter_store import AwsSsmSettingsSource

class SeerConfig(BaseSettings):
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



    default_llm_model: str = Field(default="gpt-5-mini", description="Default LLM model")

    # Taskiq / Valkey configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Valkey/Redis connection string for Taskiq broker and result backend. Use 'rediss://' for TLS/SSL connections.",
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
    # Computed Properties
    # ============================================================================

    @property
    def is_cloud_mode(self) -> bool:
        """Check if running in cloud mode."""
        return self.seer_mode == "cloud"

    @property
    def is_self_hosted(self) -> bool:
        """Check if running in self-hosted mode."""
        return self.seer_mode == "self-hosted"

    @property
    def is_clerk_configured(self) -> bool:
        """Check if Clerk authentication is configured."""
        return self.clerk_jwks_url is not None and self.clerk_issuer is not None

    @property
    def is_stripe_configured(self) -> bool:
        """Check if Stripe is configured for subscription billing."""
        return (
            self.stripe_secret_key is not None
            and self.stripe_webhook_secret is not None
        )

    @property
    def is_slack_configured(self) -> bool:
        """Check if Slack error notifications are configured."""
        return (
            self.slack_bot_token is not None
            and self.slack_error_channel_id is not None
        )

    @property
    def is_langfuse_configured(self) -> bool:
        """Check if Langfuse is configured (at least one project has credentials)."""
        if not self.langfuse_enabled:
            return False
        nexus_configured = (
            self.langfuse_nexus_public_key is not None
            and self.langfuse_nexus_secret_key is not None
        )
        workflow_configured = (
            self.langfuse_workflow_public_key is not None
            and self.langfuse_workflow_secret_key is not None
        )
        return nexus_configured or workflow_configured

    @property
    def is_langfuse_nexus_configured(self) -> bool:
        """Check if Langfuse is configured for Nexus agent tracing."""
        return (
            self.langfuse_enabled
            and self.langfuse_nexus_public_key is not None
            and self.langfuse_nexus_secret_key is not None
        )

    @property
    def is_langfuse_workflow_configured(self) -> bool:
        """Check if Langfuse is configured for Workflow tracing."""
        return (
            self.langfuse_enabled
            and self.langfuse_workflow_public_key is not None
            and self.langfuse_workflow_secret_key is not None
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
