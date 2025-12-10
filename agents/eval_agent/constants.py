"""Constants for the eval agent - imports from shared.config for consistency"""
from langgraph_sdk import get_client, get_sync_client
from langfuse import Langfuse
from shared.config import config

__all__ = [
    "LANGGRAPH_CLIENT",
    "LANGGRAPH_SYNC_CLIENT",
    "LANGFUSE_CLIENT",
]

# Local LangGraph store used for eval reflections and metadata persistence
LANGGRAPH_CLIENT = get_client(url=config.langgraph_base_url)
LANGGRAPH_SYNC_CLIENT = get_sync_client(url=config.langgraph_base_url)

# Langfuse client for tracing and observability
LANGFUSE_CLIENT = Langfuse(
    secret_key=config.langfuse_secret_key,
    host=config.langfuse_base_url
) if config.langfuse_secret_key else None
