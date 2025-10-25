from typing import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

# E2B Code Interpreter for sandbox execution
from shared.logger import get_logger
from agents.reflexion.models import ReflexionState, InputState, OutputState
from agents.reflexion.nodes.actor import actor_node
from agents.reflexion.nodes.evaluator import evaluator_node
from agents.reflexion.nodes.reflection import reflection_node

# Get logger for reflexion agent
logger = get_logger('reflexion_agent')


def finalize_node(state: ReflexionState, config: RunnableConfig) -> dict:
    """
    Finalize the result - either success or max attempts reached.
    No additional messages added - user sees natural conversation with actor only.
    """
    # Just return success flag - no additional messages
    # User only sees the natural conversation with actor
    return {
        "messages": state.trajectory
    }


# Conditional edge function
def should_continue(state: ReflexionState) -> Literal["reflect", "finalize"]:
    """
    Decide whether to continue reflection loop or finalize.
    
    Logic:
    - If evaluator passed -> finalize
    - If max attempts reached -> finalize
    - Otherwise -> reflect and try again
    """
    passed = state.evaluator_verdict.passed
    
    if passed:
        logger.info("Routing to finalize - evaluation passed")
        return "finalize"
    
    if state.current_attempt >= state.max_attempts:
        logger.info(f"Routing to finalize - max attempts ({state.max_attempts}) reached")
        return "finalize"
    
    logger.info(f"Routing to reflect - attempt {state.current_attempt}/{state.max_attempts}, continuing loop")
    return "reflect"


# Build the graph
def build_graph():
    """
    Build the reflexion agent graph with actor, evaluator, and reflection nodes.
    
    Flow:
    START -> actor -> evaluator -> [conditional]
                                    |
                                    ├─> if passed or max attempts: finalize -> END
                                    └─> if failed and attempts < max: reflection -> actor (loop)
    """
    # Create state graph
    workflow = StateGraph(ReflexionState, input=InputState, output=OutputState)
    
    # Add nodes
    workflow.add_node("actor", actor_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("finalize", finalize_node)
    
    # Add edges
    workflow.add_edge(START, "actor")
    workflow.add_edge("actor", "evaluator")
    
    # Conditional edge from evaluator
    workflow.add_conditional_edges(
        "evaluator",
        should_continue,
        {
            "reflect": "reflection",
            "finalize": "finalize"
        }
    )
    
    # Reflection loops back to actor
    workflow.add_edge("reflection", "actor")
    
    # Finalize goes to END
    workflow.add_edge("finalize", END)
    
    # Compile - LangGraph API handles persistence automatically
    graph = workflow.compile()
    
    logger.info("Reflexion graph compiled successfully")
    return graph


# Create the graph instance
graph = build_graph()

