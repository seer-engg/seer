"""Graph registry and compilation for the FastAPI server."""
from typing import Dict
from langgraph.graph import StateGraph

from api.agents.checkpointer import get_checkpointer
from shared.logger import get_logger

logger = get_logger("api.graphs")

# Registry of available graphs
_GRAPH_BUILDERS: Dict[str, callable] = {}
_COMPILED_GRAPHS: Dict[str, any] = {}


def register_graph(name: str):
    """Decorator to register a graph builder function."""
    def decorator(builder_func):
        _GRAPH_BUILDERS[name] = builder_func
        return builder_func
    return decorator


def _get_eval_agent_workflow() -> StateGraph:
    """Get the eval_agent workflow (uncompiled)."""
    from agents.eval_agent.graph import build_graph
    return build_graph()


def _get_supervisor_workflow() -> StateGraph:
    """Get the supervisor workflow (uncompiled)."""
    from agents.supervisor.graph import build_graph
    return build_graph()


# Graph builders mapping
GRAPH_BUILDERS = {
    "eval_agent": _get_eval_agent_workflow,
    "supervisor": _get_supervisor_workflow
}


async def get_compiled_graph(graph_name: str):
    """
    Get a compiled graph by name with async checkpointer.
    
    Graphs are compiled lazily and cached.
    
    Args:
        graph_name: Name of the graph (eval_agent, supervisor, codex)
        
    Returns:
        Compiled graph with checkpointer attached
        
    Raises:
        ValueError: If graph_name is not recognized
    """
    if graph_name not in GRAPH_BUILDERS:
        available = ", ".join(GRAPH_BUILDERS.keys())
        raise ValueError(f"Unknown graph: {graph_name}. Available graphs: {available}")
    
    # Check cache first
    if graph_name in _COMPILED_GRAPHS:
        return _COMPILED_GRAPHS[graph_name]
    
    # Build and compile
    logger.info(f"Building graph: {graph_name}")
    workflow = GRAPH_BUILDERS[graph_name]()
    
    # Get async checkpointer
    checkpointer = await get_checkpointer()
    
    # Compile with checkpointer
    compiled = workflow.compile(checkpointer=checkpointer)
    
    # Cache
    _COMPILED_GRAPHS[graph_name] = compiled
    logger.info(f"Graph {graph_name} compiled and cached")
    
    return compiled


def get_available_graphs() -> list[str]:
    """Get list of available graph names."""
    return list(GRAPH_BUILDERS.keys())

