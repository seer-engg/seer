from typing import List, Any, Optional
from shared.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from reflexion.agent.graph import create_reflexion
from reflexion.memory.store import Neo4jMemoryStore
from tool_hub import ToolHub

# Global memory store instance
_memory_store = None

def get_memory_store():
    global _memory_store
    if _memory_store is None:
        # OPTIMIZATION: Use Singleton pattern for DB connection
        _memory_store = Neo4jMemoryStore(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
    return _memory_store

# OPTIMIZATION: Add caching for the ephemeral agent if tools don't change often
# For now, we keep creation dynamic but ensure we don't recreate the store.
def create_ephemeral_reflexion(model, prompt, agent_id, tool_hub: Optional[ToolHub] = None, tools: Optional[List[Any]] = None, max_rounds=3):
    """
    Factory to create a reflexion graph with dynamic tools.
    
    Args:
        model: The LLM to use
        prompt: System prompt for the agent
        agent_id: Unique ID for the agent
        tool_hub: A configured ToolHub instance (optional)
        tools: List of explicit tools (optional, alternative to tool_hub)
        max_rounds: Max reflection rounds
    """
    memory = get_memory_store()
    
    # Create the graph
    graph = create_reflexion(
        model=model,
        tool_hub=tool_hub,
        tools=tools,
        prompt=prompt,
        memory_store=memory,
        agent_id=agent_id,
        max_rounds=max_rounds,
        eval_threshold=0.9
    )
    
    return graph
