"""
Type-safe configuration for Seer using Pydantic Settings.

This module provides a centralized, type-safe configuration system
that loads from environment variables and .env files.

Usage:
    from shared.config import config
    
    if score >= config.eval_pass_threshold:
        ...
"""
import os
from typing import Optional, Dict, Any
from pydantic import Field, computed_field
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
    langfuse_secret_key: Optional[str] = Field(default=None, description="Langfuse secret key for API access")
    langfuse_public_key: Optional[str] = Field(default=None, description="Langfuse public key for SDK (optional)")
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key for web search")
    github_token: Optional[str] = Field(default=None, description="GitHub token for sandbox provisioning")
    CONTEXT7_API_KEY: Optional[str] = Field(default=None, description="Context7 API key for MCP tools")
    
    # ============================================================================
    # Evaluation Agent Configuration
    # ============================================================================
    
    eval_n_rounds: int = Field(default=2, description="Number of eval rounds per version")
    eval_n_test_cases: int = Field(default=1, description="Number of test cases to generate per round")
    eval_n_versions: int = Field(default=2, description="Total versions of target agent to evaluate")
    eval_pass_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum score to pass evaluation")
    
    # LangGraph URLs
    codex_remote_url: str = Field(default="http://127.0.0.1:8003", description="URL for codex agent LangGraph")
    
    # Feature flags
    target_agent_context_level: int = Field(default=0, ge=0, le=3, description="Context level for target agent messages: 0=minimal, 1=system_goal, 2=system_goal+action, 3=full_context")
    
    langfuse_project_name: str = Field(default="target_agent", description="Langfuse project name for target agent")
    project_name: str = Field(default="eval-v1", description="Project name for metadata filtering (used in eval agent trace metadata)")
    codex_project_name: str = Field(default="codex-v1", description="Project name for metadata filtering (used in codex agent trace metadata)")

    target_agent_port: int = Field(default=2024, description="Port for target agent")
    target_agent_setup_script: str = Field(default="pip install -e .", description="Setup script for target agent")
    target_agent_command: str = Field(default="langgraph dev --host 0.0.0.0", description="Command to run target agent")
    codex_handoff_enabled: bool = Field(default=True, description="Enable handoff to codex agent")
    eval_plan_only_mode: bool = Field(default=False, description="Plan-only mode: skip execution, return after plan generation")
    eval_reasoning_effort: str = Field(default="medium", description="Reasoning effort for eval agent planning: 'minimal', 'medium', or 'high'")
    
    # Base template for E2B sandbox
    base_template_alias: str = Field(default="seer-base", description="E2B template alias")


    # ============================================================================
    # Codex Agent Configuration
    # ============================================================================
    allow_pr: bool = Field(default=True, description="Allow PR creation even if eval  fails")
    eval_agent_handoff_enabled: bool = Field(default=False, description="Enable handoff to eval agent")
    codex_reasoning_effort: str = Field(default="high", description="Reasoning effort for Codex developer node: 'minimal', 'medium', or 'high'")
    
    # ============================================================================
    # Neo4j Graph Database Configuration
    # ============================================================================
    
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: Optional[str] = Field(default=None, description="Neo4j username")
    neo4j_password: Optional[str] = Field(default=None, description="Neo4j password")
    
    # ============================================================================
    # LangGraph Checkpointer Configuration
    # ============================================================================
    
    database_uri: Optional[str] = Field(
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
    
    # Eval reflections index
    eval_reflections_index_name: str = Field(default="eval_reflections", description="Neo4j index name for eval reflections")
    eval_reflections_node_label: str = Field(default="EvalReflection", description="Neo4j node label for reflections")
    eval_reflections_embedding_property: str = Field(default="embedding", description="Property name for embeddings")
    
    # MCP tools index
    tool_node_label: str = Field(default="MCPTool", description="Neo4j node label for tools")
    tool_embed_prop: str = Field(default="embedding", description="Property name for tool embeddings")
    tool_vector_index: str = Field(default="mcp_tools_index", description="Neo4j index name for tools")
    tool_hub_index_dir: str = Field(default="tool_hub_index", description="Directory for ToolHub index")

    # user
    user_id: str = Field(default="", description="User ID")
    github_default_owner: str = Field(default="", description="GitHub default owner")
    github_default_repo: str = Field(default="", description="GitHub default repo")
    
    # ============================================================================
    # MCP (Model Context Protocol) Configuration
    # ============================================================================
    
    # Composio configuration
    composio_api_key: Optional[str] = Field(default=None, description="Composio API key (if required)")
    
    # ============================================================================
    # Langfuse Configuration
    # ============================================================================
    
    langfuse_base_url: str = Field(default="http://localhost:3000", description="Langfuse host URL (self-hosted instance)")
    
    # ============================================================================
    # Asana Configuration
    # ============================================================================
    
    asana_workspace_id: Optional[str] = Field(default=None, description="Asana workspace ID")
    asana_team_gid: Optional[str] = Field(default=None, description="Asana default team GID")
    asana_project_id: Optional[str] = Field(default=None, description="Asana project ID to reuse (for free plans without teams)")
    # ============================================================================
    # Computed Properties
    # ============================================================================

    default_llm_model: str = Field(default="gpt-5-mini", description="Default LLM model")
    
    @computed_field
    @property
    def target_agent_envs(self) -> Dict[str, Any]:
        """Environment variables for target agent."""
        envs: Dict[str, Any] = {}
        if self.openai_api_key:
            envs["OPENAI_API_KEY"] = self.openai_api_key
        if self.composio_api_key:
            envs["COMPOSIO_API_KEY"] = self.composio_api_key
        # Add Langfuse environment variables if configured
        if self.langfuse_secret_key:
            envs['LANGFUSE_SECRET_KEY'] = self.langfuse_secret_key
        if self.langfuse_public_key:
            envs['LANGFUSE_PUBLIC_KEY'] = self.langfuse_public_key
        if self.langfuse_base_url:
            envs['LANGFUSE_BASE_URL'] = self.langfuse_base_url
        if self.langfuse_project_name:
            envs['LANGFUSE_PROJECT_NAME'] = self.langfuse_project_name
        return envs

    
    @property
    def is_langfuse_configured(self) -> bool:
        """Check if Langfuse is configured."""
        return self.langfuse_secret_key is not None and self.langfuse_base_url is not None
# ============================================================================
# Global Config Instance
# ============================================================================

config = SeerConfig()


