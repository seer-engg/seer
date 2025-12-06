from typing import Annotated, Optional
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
import logging
import uuid

logger = logging.getLogger(__name__)

NAMESPACE = ("artifacts",)

@tool
async def read_memory(key: str, store: Annotated[BaseStore, InjectedStore]) -> str:
    """
    Reads a memory/artifact from the shared store by key.
    Useful for retrieving large data passed by the supervisor or other workers.
    """
    try:
        item = await store.aget(NAMESPACE, key)
        if item:
            data = item.value.get("data")
            return str(data)
        return f"Error: No memory found for key '{key}'"
    except Exception as e:
        logger.error(f"Error reading memory {key}: {e}")
        return f"Error reading memory: {e}"

@tool
async def write_memory(key: str, content: str, store: Annotated[BaseStore, InjectedStore]) -> str:
    """
    Saves a large memory/artifact to the shared store.
    ALWAYS use this for large API responses or datasets instead of putting them in the message.
    Returns the key to reference this data.
    """
    try:
        # If key is generic/empty or "auto", generate one
        if not key or key == "auto":
            key = str(uuid.uuid4())[:8]
            
        await store.aput(NAMESPACE, key, {"data": content})
        logger.info(f"💾 Saved artifact to memory: {key} (Size: {len(content)} chars)")
        return f"Saved to memory key: {key}"
    except Exception as e:
        logger.error(f"Error writing memory {key}: {e}")
        return f"Error writing memory: {e}"

