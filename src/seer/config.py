"""
Type-safe configuration for Seer using Pydantic Settings.

This module provides a centralized, type-safe configuration system
that loads from environment variables and .env files.

Usage:
    from seer.config import config

    if score >= config.eval_pass_threshold:
        ...
"""
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ============================================================================
    # API Keys & Authentication
    # ============================================================================

    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key for LLM and embeddings"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key for Claude models"
    )
    tavily_api_key: Optional[str] = Field(
        default=None, description="Tavily API key for web search"
    )
    github_token: Optional[str] = Field(
        default=None, description="GitHub token for sandbox provisioning"
    )

    # ============================================================================
    # LangGraph Checkpointer Configuration
    # ============================================================================

    DATABASE_URL: Optional[str] = Field(
        default=None,
        description=(
            "PostgreSQL connection string for LangGraph checkpointer "
            "(e.g., postgresql://user:pass@host:port/db). "
            "Required for human-in-the-loop interrupts."
        )
    )

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

    default_llm_model: str = Field(default="gpt-5-mini", description="Default LLM model")

    # Taskiq / Redis configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string for Taskiq broker and result backend",
    )

    # Tool index configuration
    tool_index_path: str = Field(
        default="./data/tool_index", description="Path to store tool vector index"
    )
    tool_index_auto_generate: bool = Field(
        default=True, description="Auto-generate tool index on startup if missing"
    )

    GOOGLE_CLIENT_ID: str = Field(default="", description="Google OAuth client ID")
    GOOGLE_CLIENT_SECRET: str = Field(default="", description="Google OAuth client secret")

    GITHUB_CLIENT_ID: Optional[str] = Field(
        default=None, description="GitHub OAuth client ID"
    )
    GITHUB_CLIENT_SECRET: Optional[str] = Field(
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

    FRONTEND_URL:str = Field(
        default="http://localhost:5173", description="Frontend application URL"
    )

    # ============================================================================
    # Feature Flags
    # ============================================================================

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
    # PostHog Analytics Configuration
    # ============================================================================

    posthog_api_key: Optional[str] = Field(
        default="phc_9s65auHWk9fXqXBEHA1x53FIuMKGurVOSF2ZfgfCWT2",
        description="PostHog API key for analytics tracking"
    )
    posthog_host: Optional[str] = Field(
        default="https://us.i.posthog.com",
        description="PostHog host URL (e.g., https://app.posthog.com or self-hosted)"
    )
    posthog_enabled: bool = Field(
        default=True,
        description="Enable PostHog analytics and error tracking"
    )
    posthog_opt_out: bool = Field(
        default=False,
        description="Opt-out of all telemetry and analytics (privacy mode)"
    )
    posthog_error_sampling_rate: float = Field(
        default=1.0,
        description="Error event sampling rate (0.0 to 1.0)"
    )
    posthog_filter_sensitive_data: bool = Field(
        default=True,
        description="Filter PII and secrets from analytics events"
    )

    webhook_base_url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL for webhook callbacks (e.g., https://seer.example.com). "
            "Used by external services to send webhooks. "
            "Defaults to http://localhost:8000 if not set."
        ),
    )
    REDIRECT_URI_SCHEME: str = Field(
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
    stripe_webhook_secret: Optional[str] = Field(
        default=None,
        description="Stripe webhook signing secret (whsec_...)"
    )
    clerk_secret_key: Optional[str] = Field(
        default=None,
        description="Clerk secret key for updating user metadata"
    )
    frontend_url: str = Field(
        default="http://localhost:5173",
        description="Frontend URL for Stripe checkout redirects"
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
    def is_posthog_configured(self) -> bool:
        """Check if PostHog is configured and enabled."""
        return (
            self.posthog_enabled
            and not self.posthog_opt_out
            and self.posthog_api_key is not None
            and self.posthog_host is not None
        )

    @property
    def is_stripe_configured(self) -> bool:
        """Check if Stripe is configured for subscription billing."""
        return (
            self.stripe_secret_key is not None
            and self.stripe_webhook_secret is not None
        )


# ============================================================================
# Global Config Instance
# ============================================================================

config = SeerConfig()
