"""Constants for the eval agent - imports from shared.config for consistency"""
from langgraph_sdk import get_client, get_sync_client
from langsmith import Client
from shared.config import config

__all__ = [
    "LANGGRAPH_CLIENT",
    "LANGGRAPH_SYNC_CLIENT",
    "LANGSMITH_CLIENT",
]

# Local LangGraph store used for eval reflections and metadata persistence
LANGGRAPH_CLIENT = get_client(url=config.langgraph_base_url)
LANGGRAPH_SYNC_CLIENT = get_sync_client(url=config.langgraph_base_url)

# LangSmith client for experiment uploads and remote graph execution
LANGSMITH_CLIENT = Client(api_key=config.langsmith_api_key)
