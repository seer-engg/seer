"""
Type-safe configuration for Seer using Pydantic Settings.

This module provides a centralized, type-safe configuration system
that loads from environment variables and .env files.

Usage:
    from shared.config import config
    
    if score >= config.eval_pass_threshold:
        ...
"""
from typing import Optional, Dict, Any
from pydantic import Field, computed_field, model_validator
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
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key for LLM and embeddings")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key for Claude models")
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key for web search")
    github_token: Optional[str] = Field(default=None, description="GitHub token for sandbox provisioning")
    CONTEXT7_API_KEY: Optional[str] = Field(default=None, description="Context7 API key for MCP tools")
    

    target_agent_port: int = Field(default=2024, description="Port for target agent")
    target_agent_command: str = Field(default="langgraph dev --host 0.0.0.0", description="Command to run target agent")

    
    # Base template for E2B sandbox
    base_template_alias: str = Field(default="seer-base", description="E2B template alias")
    
    # ============================================================================
    # LangGraph Checkpointer Configuration
    # ============================================================================
    
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string for LangGraph checkpointer (e.g., postgresql://user:pass@host:port/db). Required for human-in-the-loop interrupts."
    )
    
    # ============================================================================
    # PostgreSQL Tool Autonomy Configuration
    # ============================================================================
    
    postgres_write_requires_approval: bool = Field(
        default=True,
        description="If True, PostgreSQL write operations (INSERT, UPDATE, DELETE, DDL) require human approval via interrupt before execution. Read operations are always allowed."
    )
    
    # Vector embeddings configuration
    embedding_dims: int = Field(default=1536, description="OpenAI embedding dimensions")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    embedding_batch_size: int = Field(default=128, description="OpenAI embedding batch size")
    
    
    # ============================================================================
    # MLflow Configuration
    # ============================================================================
    
    mlflow_tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking server URI (e.g., http://localhost:5000)")
    mlflow_experiment_name: Optional[str] = Field(default=None, description="MLflow experiment name for organizing runs")
    
    
    # ============================================================================
    # Deployment Mode Configuration
    # ============================================================================
    
    seer_mode: str = Field(default="self-hosted", description="Deployment mode: 'self-hosted' or 'cloud'")
    
    # ============================================================================
    # Clerk Authentication Configuration
    # ============================================================================
    
    clerk_jwks_url: Optional[str] = Field(default=None, description="Clerk JWKS URL for JWT verification")
    clerk_issuer: Optional[str] = Field(default=None, description="Clerk JWT issuer (e.g., https://clerk.your-domain.com)")
    clerk_audience: Optional[str] = Field(default=None, description="Clerk JWT audience (e.g., ['api.your-domain.com'])")
    
    default_llm_model: str = Field(default="gpt-5-mini", description="Default LLM model")
    
    # Tool index configuration
    tool_index_path: str = Field(default="./data/tool_index", description="Path to store tool vector index")
    tool_index_auto_generate: bool = Field(default=True, description="Auto-generate tool index on startup if missing")

    GOOGLE_CLIENT_ID: str = Field(default="", description="Google OAuth client ID")
    GOOGLE_CLIENT_SECRET: str = Field(default="", description="Google OAuth client secret")
    
    GITHUB_CLIENT_ID: Optional[str] = Field(default=None, description="GitHub OAuth client ID")
    GITHUB_CLIENT_SECRET: Optional[str] = Field(default=None, description="GitHub OAuth client secret")
    GITHUB_MCP_SERVER_URL: Optional[str] = Field(
        default="https://api.githubcopilot.com/mcp/",
        description="GitHub MCP server URL (for streamable HTTP transport, e.g., http://localhost:8080/mcp)"
    )
    # ============================================================================
    # Computed Properties
    # ============================================================================



    
    @computed_field
    @property
    def target_agent_envs(self) -> Dict[str, Any]:
        """Environment variables for target agent."""
        envs: Dict[str, Any] = {}
        if self.openai_api_key:
            envs["OPENAI_API_KEY"] = self.openai_api_key
        return envs

    
    @property
    def is_mlflow_configured(self) -> bool:
        """Check if MLflow is configured."""
        return self.mlflow_tracking_uri is not None
    
    @property
    def is_mlflow_tracing_enabled(self) -> bool:
        """Check if MLflow tracing is enabled."""
        return self.is_mlflow_configured
    
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

# ============================================================================
# Global Config Instance
# ============================================================================

config = SeerConfig()


