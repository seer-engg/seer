"""
Central configuration for Seer.

This module consolidates all configuration constants from various modules.
It provides a single source of truth for configuration values.
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# API Keys & Authentication
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ============================================================================
# Evaluation Agent Configuration
# ============================================================================

# Evaluation loop thresholds
N_ROUNDS = int(os.getenv("EVAL_N_ROUNDS", "2"))
N_TEST_CASES = int(os.getenv("EVAL_N_TEST_CASES", "1"))  # Number of test cases to generate per round
N_VERSIONS = int(os.getenv("EVAL_N_VERSIONS", "2"))  # Total versions of the target agent
EVAL_PASS_THRESHOLD = float(os.getenv("EVAL_PASS_THRESHOLD", "0.8"))

# LangGraph URLs
LANGGRAPH_BASE_URL = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:8002")
CODEX_REMOTE_URL = os.getenv("CODEX_REMOTE_URL", "http://127.0.0.1:8003")
EVAL_REMOTE_URL = os.getenv("EVAL_REMOTE_URL", "http://127.0.0.1:8002")

# Feature flags
CODEX_HANDOFF_ENABLED = os.getenv("CODEX_HANDOFF_ENABLED", "false").lower() == "true"
USE_GENETIC_TEST_GENERATION = os.getenv("USE_GENETIC_TEST_GENERATION", "false").lower() == "true"
USE_AGENTIC_TEST_GENERATION = os.getenv("USE_AGENTIC_TEST_GENERATION", "false").lower() == "true"
EVAL_AGENT_LOAD_DEFAULT_MCPS = os.getenv("EVAL_AGENT_LOAD_DEFAULT_MCPS", "true").lower() == "true"


# ============================================================================
# Sandbox & Target Agent Configuration
# ============================================================================

TARGET_AGENT_LANGSMITH_PROJECT = 'target_agent'
TARGET_AGENT_PORT = 2024
TARGET_AGENT_SETUP_SCRIPT = "pip install -e ."
TARGET_AGENT_COMMAND = "langgraph dev --host 0.0.0.0"

# Base template for E2B sandbox
BASE_TEMPLATE_ALIAS = "seer-base"
BASE_TEMPLATE_CPU_COUNT = 1
BASE_TEMPLATE_MEMORY_MB = 1024

# Target agent environment variables
TARGET_AGENT_ENVS = {
    'OPENAI_API_KEY': OPENAI_API_KEY,
    'LANGSMITH_API_KEY': LANGSMITH_API_KEY,
    'LANGSMITH_PROJECT': TARGET_AGENT_LANGSMITH_PROJECT,
}


# ============================================================================
# Neo4j Graph Database Configuration
# ============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Vector embeddings configuration
EMBEDDING_DIMS = 1536  # OpenAI embeddings dimension

# Eval reflections index
EVAL_REFLECTIONS_INDEX_NAME = "eval_reflections"
EVAL_REFLECTIONS_NODE_LABEL = "EvalReflection"
EVAL_REFLECTIONS_EMBEDDING_PROPERTY = "embedding"

# MCP tools index
TOOL_NODE_LABEL = "MCPTool"
TOOL_EMBED_PROP = "embedding"
TOOL_VECTOR_INDEX = "mcp_tools_index"


# ============================================================================
# MCP (Model Context Protocol) Configuration
# ============================================================================

# Default MCP services to load
DEFAULT_MCP_SERVICES = ["asana", "github", "langchain_docs"]

# MCP service URLs
LANGCHAIN_MCP_URL = "https://docs.langchain.com/mcp"


# ============================================================================
# LangSmith Configuration (for compatibility)
# ============================================================================

LANGSMITH_API_URL = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")


# ============================================================================
# Asana Configuration
# ============================================================================

ASANA_WORKSPACE_ID = os.getenv("ASANA_WORKSPACE_ID")
ASANA_DEFAULT_WORKSPACE_GID = os.getenv("ASANA_DEFAULT_WORKSPACE_GID")
ASANA_PROJECT_ID = os.getenv("ASANA_PROJECT_ID")
ASANA_DEFAULT_PROJECT_GID = os.getenv("ASANA_DEFAULT_PROJECT_GID")


# ============================================================================
# Helper Functions
# ============================================================================

def get_asana_workspace_gid() -> Optional[str]:
    """Get Asana workspace GID from environment."""
    return ASANA_WORKSPACE_ID or ASANA_DEFAULT_WORKSPACE_GID


def get_asana_project_gid() -> Optional[str]:
    """Get Asana project GID from environment."""
    return ASANA_PROJECT_ID or ASANA_DEFAULT_PROJECT_GID

