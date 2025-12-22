"""
Configuration and environment variable management.

This module MUST be imported before any other modules that depend on environment variables.
It loads and validates all required environment variables upfront.
"""
import os
from dotenv import load_dotenv

# Load environment variables FIRST (before any other imports)
# This works for both .env files (local) and Railway's environment variables (production)
load_dotenv()

# Required environment variables
_REQUIRED_VARS = {
    "OPENAI_API_KEY": "OpenAI API key for LLM calls",
    "PINECONE_API_KEY": "Pinecone API key for vector search",
    "PINECONE_INDEX_NAME": "Pinecone index name for tool search",
}

def _validate_environment():
    """Validate that all required environment variables are set."""
    missing = []
    for var_name, description in _REQUIRED_VARS.items():
        if not os.getenv(var_name):
            missing.append(f"{var_name} ({description})")
    
    if missing:
        error_msg = (
            "Missing required environment variables:\n"
            + "\n".join(f"  - {var}" for var in missing)
            + "\n\n"
            + "Please set these in Railway Variables or your .env file."
        )
        raise ValueError(error_msg)

# Validate environment variables immediately when this module is imported
_validate_environment()

# Export validated environment variables as module-level constants
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")  # type: ignore
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")  # type: ignore
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME")  # type: ignore

def get_env_summary() -> dict:
    """Get a summary of environment variable status (for health checks)."""
    return {
        "required": {
            var: bool(os.getenv(var)) for var in _REQUIRED_VARS.keys()
        },
        "optional": {}
    }

