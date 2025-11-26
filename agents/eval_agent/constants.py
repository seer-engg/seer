"""Constants for the eval agent - imports from shared.config for consistency"""
from langgraph_sdk import get_client, get_sync_client
from langsmith import Client
from shared.config import (
    OPENAI_API_KEY,
    LANGSMITH_API_KEY,
    LANGGRAPH_BASE_URL,
    CODEX_REMOTE_URL,
    EVAL_REMOTE_URL,
    N_ROUNDS,
    N_TEST_CASES,
    N_VERSIONS,
    EVAL_PASS_THRESHOLD,
)

# Re-export for backward compatibility
__all__ = [
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGGRAPH_BASE_URL",
    "CODEX_REMOTE_URL",
    "EVAL_REMOTE_URL",
    "N_ROUNDS",
    "N_TEST_CASES",
    "N_VERSIONS",
    "EVAL_PASS_THRESHOLD",
    "LANGGRAPH_CLIENT",
    "LANGGRAPH_SYNC_CLIENT",
    "LANGSMITH_CLIENT",
]

# Local LangGraph store used for eval reflections and metadata persistence
LANGGRAPH_CLIENT = get_client(url=LANGGRAPH_BASE_URL)
LANGGRAPH_SYNC_CLIENT = get_sync_client(url=LANGGRAPH_BASE_URL)

# LangSmith client for experiment uploads and remote graph execution
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)
