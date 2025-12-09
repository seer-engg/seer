"""
Factory for creating Supervisor-style reflection agents.
Replaces Reflexion with simple ReAct agent pattern.
"""
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from shared.llm import get_llm_without_responses_api
from shared.logger import get_logger

logger = get_logger("eval_agent.reflect.agent_factory")

def create_reflection_agent(tools, system_prompt):
    """
    Create a Supervisor-style reflection agent.
    
    Uses create_agent() pattern (same as Supervisor), no memory store.
    This replaces create_ephemeral_reflexion().
    
    Args:
        tools: List of tools for the agent
        system_prompt: System prompt for the agent
        
    Returns:
        Agent runnable (same interface as create_agent)
    """
    llm = get_llm_without_responses_api()
    
    middleware = [
        ToolCallLimitMiddleware(thread_limit=30, run_limit=10),
    ]
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware
    )
    
    logger.info("Created reflection agent using Supervisor pattern")
    return agent
