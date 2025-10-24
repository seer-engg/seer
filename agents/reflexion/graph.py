import json
import os
import re
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool, ToolRuntime
from langchain_core.tools import InjectedToolCallId

# E2B Code Interpreter for sandbox execution
from shared.llm import get_llm
from shared.logger import get_logger
from agents.reflexion.models import ReflexionState
from agents.reflexion.nodes.actor import actor_node
from agents.reflexion.nodes.evaluator import evaluator_node
from agents.reflexion.nodes.reflection import reflection_node
# Get logger for reflexion agent
logger = get_logger('reflexion_agent')

# Memory store namespace for reflexion feedback
MEMORY_NAMESPACE = ("reflexion", "feedback")


def finalize_node(state: ReflexionState, config: RunnableConfig) -> dict:
    """
    Finalize the result - either success or max attempts reached.
    No additional messages added - user sees natural conversation with actor only.
    """
    current_attempt = state.get("current_attempt", 1)
    max_attempts = state.get("max_attempts", 3)
    verdict_dict = state.get("evaluator_verdict", {})
    passed = verdict_dict.get("passed", False)
    
    if passed:
        logger.info(f"✅ Success! Response passed evaluation on attempt {current_attempt}/{max_attempts}")
    else:
        logger.info(f"⚠️ Max attempts ({max_attempts}) reached. Final Score: {verdict_dict.get('score', 0.0)}")
    
    # Just return success flag - no additional messages
    # User only sees the natural conversation with actor
    return {
        "success": passed
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
    verdict_dict = state.get("evaluator_verdict", {})
    passed = verdict_dict.get("passed", False)
    current_attempt = state.get("current_attempt", 1)
    max_attempts = state.get("max_attempts", 3)
    
    if passed:
        logger.info("Routing to finalize - evaluation passed")
        return "finalize"
    
    if current_attempt >= max_attempts:
        logger.info(f"Routing to finalize - max attempts ({max_attempts}) reached")
        return "finalize"
    
    logger.info(f"Routing to reflect - attempt {current_attempt}/{max_attempts}, continuing loop")
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
    workflow = StateGraph(ReflexionState)
    
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

