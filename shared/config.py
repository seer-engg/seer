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
from typing import Optional, List, Dict, Any
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
    codex_handoff_enabled: bool = Field(default=False, description="Enable handoff to codex agent")
    use_genetic_test_generation: bool = Field(default=False, description="Use genetic algorithm for test generation")
    use_agentic_test_generation: bool = Field(default=False, description="Use agentic approach for test generation")
    eval_agent_load_default_mcps: bool = Field(default=True, description="Load default MCP services")
    
    # ============================================================================
    # Sandbox & Target Agent Configuration
    # ============================================================================
    
    target_agent_langsmith_project: str = Field(default="target_agent", description="LangSmith project for target agent")
    target_agent_port: int = Field(default=2024, description="Port for target agent")
    target_agent_setup_script: str = Field(default="pip install -e .", description="Setup script for target agent")
    target_agent_command: str = Field(default="langgraph dev --host 0.0.0.0", description="Command to run target agent")
    
    # Base template for E2B sandbox
    base_template_alias: str = Field(default="seer-base", description="E2B template alias")
    base_template_cpu_count: int = Field(default=1, description="CPU count for sandbox")
    base_template_memory_mb: int = Field(default=1024, description="Memory in MB for sandbox")
    
    # ============================================================================
    # Neo4j Graph Database Configuration
    # ============================================================================
    
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: Optional[str] = Field(default=None, description="Neo4j username")
    neo4j_password: Optional[str] = Field(default=None, description="Neo4j password")
    
    # Vector embeddings configuration
    embedding_dims: int = Field(default=1536, description="OpenAI embedding dimensions")
    
    # Eval reflections index
    eval_reflections_index_name: str = Field(default="eval_reflections", description="Neo4j index name for eval reflections")
    eval_reflections_node_label: str = Field(default="EvalReflection", description="Neo4j node label for reflections")
    eval_reflections_embedding_property: str = Field(default="embedding", description="Property name for embeddings")
    
    # MCP tools index
    tool_node_label: str = Field(default="MCPTool", description="Neo4j node label for tools")
    tool_embed_prop: str = Field(default="embedding", description="Property name for tool embeddings")
    tool_vector_index: str = Field(default="mcp_tools_index", description="Neo4j index name for tools")
    
    # ============================================================================
    # MCP (Model Context Protocol) Configuration
    # ============================================================================
    
    langchain_mcp_url: str = Field(default="https://docs.langchain.com/mcp", description="LangChain MCP documentation URL")
    
    # ============================================================================
    # LangSmith Configuration
    # ============================================================================
    
    langsmith_api_url: str = Field(default="https://api.smith.langchain.com", description="LangSmith API base URL")
    
    # ============================================================================
    # Asana Configuration
    # ============================================================================
    
    asana_workspace_id: Optional[str] = Field(default=None, description="Asana workspace ID")
    asana_default_workspace_gid: Optional[str] = Field(default=None, description="Asana default workspace GID")
    asana_project_id: Optional[str] = Field(default=None, description="Asana project ID")
    asana_default_project_gid: Optional[str] = Field(default=None, description="Asana default project GID")
    
    # ============================================================================
    # Computed Properties
    # ============================================================================
    
    @computed_field
    @property
    def target_agent_envs(self) -> Dict[str, Any]:
        """Environment variables for target agent."""
        return {
            'OPENAI_API_KEY': self.openai_api_key,
            'LANGSMITH_API_KEY': self.langsmith_api_key,
            'LANGSMITH_PROJECT': self.target_agent_langsmith_project,
        }
    
    def get_asana_workspace_gid(self) -> Optional[str]:
        """Get Asana workspace GID from environment."""
        return self.asana_workspace_id or self.asana_default_workspace_gid
    
    def get_asana_project_gid(self) -> Optional[str]:
        """Get Asana project GID from environment."""
        return self.asana_project_id or self.asana_default_project_gid


# ============================================================================
# Global Config Instance
# ============================================================================

config = SeerConfig()


# ============================================================================
# Backward Compatibility Exports (ALL_CAPS style)
# These are kept for backward compatibility with existing code.
# New code should use `config.field_name` instead of `FIELD_NAME`.
# ============================================================================

# API Keys
OPENAI_API_KEY = config.openai_api_key
LANGSMITH_API_KEY = config.langsmith_api_key
TAVILY_API_KEY = config.tavily_api_key

# Eval Configuration
N_ROUNDS = config.eval_n_rounds
N_TEST_CASES = config.eval_n_test_cases
N_VERSIONS = config.eval_n_versions
EVAL_PASS_THRESHOLD = config.eval_pass_threshold

# URLs
LANGGRAPH_BASE_URL = config.langgraph_base_url
CODEX_REMOTE_URL = config.codex_remote_url
EVAL_REMOTE_URL = config.eval_remote_url

# Feature Flags
CODEX_HANDOFF_ENABLED = config.codex_handoff_enabled
USE_GENETIC_TEST_GENERATION = config.use_genetic_test_generation
USE_AGENTIC_TEST_GENERATION = config.use_agentic_test_generation
EVAL_AGENT_LOAD_DEFAULT_MCPS = config.eval_agent_load_default_mcps

# Target Agent
TARGET_AGENT_LANGSMITH_PROJECT = config.target_agent_langsmith_project
TARGET_AGENT_PORT = config.target_agent_port
TARGET_AGENT_SETUP_SCRIPT = config.target_agent_setup_script
TARGET_AGENT_COMMAND = config.target_agent_command
TARGET_AGENT_ENVS = config.target_agent_envs

# Sandbox
BASE_TEMPLATE_ALIAS = config.base_template_alias
BASE_TEMPLATE_CPU_COUNT = config.base_template_cpu_count
BASE_TEMPLATE_MEMORY_MB = config.base_template_memory_mb

# Neo4j
NEO4J_URI = config.neo4j_uri
NEO4J_USERNAME = config.neo4j_username
NEO4J_PASSWORD = config.neo4j_password
EMBEDDING_DIMS = config.embedding_dims
EVAL_REFLECTIONS_INDEX_NAME = config.eval_reflections_index_name
EVAL_REFLECTIONS_NODE_LABEL = config.eval_reflections_node_label
EVAL_REFLECTIONS_EMBEDDING_PROPERTY = config.eval_reflections_embedding_property
TOOL_NODE_LABEL = config.tool_node_label
TOOL_EMBED_PROP = config.tool_embed_prop
TOOL_VECTOR_INDEX = config.tool_vector_index

# MCP
LANGCHAIN_MCP_URL = config.langchain_mcp_url

# LangSmith
LANGSMITH_API_URL = config.langsmith_api_url

# Asana
ASANA_WORKSPACE_ID = config.asana_workspace_id
ASANA_DEFAULT_WORKSPACE_GID = config.asana_default_workspace_gid
ASANA_PROJECT_ID = config.asana_project_id
ASANA_DEFAULT_PROJECT_GID = config.asana_default_project_gid


# ============================================================================
# Helper Functions (for backward compatibility)
# ============================================================================

def get_asana_workspace_gid() -> Optional[str]:
    """Get Asana workspace GID from environment."""
    return config.get_asana_workspace_gid()


def get_asana_project_gid() -> Optional[str]:
    """Get Asana project GID from environment."""
    return config.get_asana_project_gid()
