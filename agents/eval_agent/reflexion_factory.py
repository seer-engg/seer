from typing import List, Any, Dict, Optional
from langchain_core.tools import BaseTool
from shared.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from reflexion.core.agent.graph import create_reflexion
from reflexion.core.memory.store import Neo4jMemoryStore
from tool_hub import ToolHub

# Global memory store instance
_memory_store = None

def get_memory_store():
    global _memory_store
    if _memory_store is None:
        _memory_store = Neo4jMemoryStore(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
    return _memory_store

def create_ephemeral_reflexion(model, tool_hub: ToolHub, prompt, agent_id, max_rounds=3):
    """
    Factory to create a reflexion graph with dynamic tools.
    
    Args:
        model: The LLM to use
        tool_hub: A configured ToolHub instance
        prompt: System prompt for the agent
        agent_id: Unique ID for the agent
        max_rounds: Max reflection rounds
    """
    memory = get_memory_store()
    
    # Create the graph
    graph = create_reflexion(
        model=model,
        tool_hub=tool_hub,
        prompt=prompt,
        memory_store=memory,
        agent_id=agent_id,
        max_rounds=max_rounds,
        eval_threshold=0.9
    )
    
    return graph
