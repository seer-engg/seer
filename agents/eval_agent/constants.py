"""Constants for the eval agent - imports from shared.config for consistency"""
from langfuse import Langfuse
from shared.config import config

__all__ = [
    "LANGFUSE_CLIENT",
]


# Langfuse client for tracing and observability
LANGFUSE_CLIENT = Langfuse(
    secret_key=config.langfuse_secret_key,
    host=config.langfuse_base_url
) if config.langfuse_secret_key else None
