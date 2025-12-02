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
    
    openai_api_key: str = Field(description="OpenAI API key for LLM and embeddings")
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key for tracing")
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
    langgraph_base_url: str = Field(default="http://127.0.0.1:8002", description="Base URL for eval agent LangGraph")
    codex_remote_url: str = Field(default="http://127.0.0.1:8003", description="URL for codex agent LangGraph")
    eval_remote_url: str = Field(default="http://127.0.0.1:8002", description="URL for eval agent LangGraph")
    
    # Feature flags
    eval_agent_load_default_mcps: bool = Field(default=True, description="Load default MCP services")
    
    target_agent_langsmith_project: str = Field(default="target_agent", description="LangSmith project for target agent")
    target_agent_port: int = Field(default=2024, description="Port for target agent")
    target_agent_setup_script: str = Field(default="pip install -e .", description="Setup script for target agent")
    target_agent_command: str = Field(default="langgraph dev --host 0.0.0.0", description="Command to run target agent")
    codex_handoff_enabled: bool = Field(default=True, description="Enable handoff to codex agent")
    
    # Base template for E2B sandbox
    base_template_alias: str = Field(default="seer-base", description="E2B template alias")


    # ============================================================================
    # Codex Agent Configuration
    # ============================================================================
    allow_pr: bool = Field(default=True, description="Allow PR creation even if eval  fails")
    eval_agent_handoff_enabled: bool = Field(default=False, description="Enable handoff to eval agent")
    
    # ============================================================================
    # Neo4j Graph Database Configuration
    # ============================================================================
    
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: Optional[str] = Field(default=None, description="Neo4j username")
    neo4j_password: Optional[str] = Field(default=None, description="Neo4j password")
    
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
    composio_user_id: Optional[str] = Field(default=None, description="Composio user ID for tool access")
    composio_api_key: Optional[str] = Field(default=None, description="Composio API key (if required)")
    
    # ============================================================================
    # LangSmith Configuration
    # ============================================================================
    
    langsmith_api_url: str = Field(default="https://api.smith.langchain.com", description="LangSmith API URL")
    
    # ============================================================================
    # Asana Configuration
    # ============================================================================
    
    asana_workspace_id: Optional[str] = Field(default=None, description="Asana workspace ID")
    asana_team_gid: Optional[str] = Field(default=None, description="Asana default team GID")
    # ============================================================================
    # Computed Properties
    # ============================================================================

    default_llm_model: str = Field(default="gpt-5.1", description="Default LLM model")
    
    @computed_field
    @property
    def target_agent_envs(self) -> Dict[str, Any]:
        """Environment variables for target agent."""
        return {
            'OPENAI_API_KEY': self.openai_api_key,
            'LANGSMITH_API_KEY': self.langsmith_api_key,
            'LANGSMITH_PROJECT': self.target_agent_langsmith_project,
            'COMPOSIO_USER_ID': self.composio_user_id,
            'COMPOSIO_API_KEY': self.composio_api_key,
        }
    
    def get_asana_workspace_gid(self) -> Optional[str]:
        """Get Asana workspace GID from environment."""
        return self.asana_workspace_id 


# ============================================================================
# Global Config Instance
# ============================================================================

config = SeerConfig()


# ============================================================================
# Helper Functions (for backward compatibility)
# ============================================================================

def get_asana_workspace_gid() -> Optional[str]:
    """Get Asana workspace GID from environment."""
    return config.get_asana_workspace_gid()
