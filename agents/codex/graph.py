"""Codex graph"""
from __future__ import annotations
from langgraph.graph import END, START, StateGraph
from shared.logger import get_logger
from shared.schema import CodexInput, CodexOutput
from agents.codex.state import CodexState
from agents.codex.nodes.raise_pr import raise_pr
from agents.codex.nodes.deploy import deploy_service
from agents.codex.nodes.test_server_ready import test_server_ready
from agents.codex.nodes import (
    planner, coder, evaluator, reflector, finalize,
    initialize_project
)
import os

logger = get_logger("codex.graph")


def is_server_ready(state: CodexState) -> CodexState:
    """This router checks if the server is ready at the *start*"""
    if state.server_running:
        return "planner"
    else:
        # If server fails initial check, we can't proceed.
        logger.error("Initial server readiness check failed. Ending graph.")
        return "end"

def should_reflect_or_raise_pr(state: CodexState) -> CodexState:
    """This router decides what to do *after* an implementation attempt"""
    if state.success:
        logger.info("Implementation successful. Proceeding to raise PR.")
        return "raise-pr"
    
    if state.attempt_number >= state.max_attempts:
        logger.warning(f"Max attempts ({state.max_attempts}) reached. Ending run.")
        if os.getenv("ALLOW_PR") == "false":            
            return "end"
        else:
            return "raise-pr"
        
    logger.info(f"Implementation failed (Attempt {state.attempt_number}). Reflecting.")
    return "reflector"

def is_codeer_implementation_working(state: CodexState) -> CodexState:
    """This router checks if the codeer implementation is working"""
    if state.server_running:
        return "evaluator"
    else:
        return "coder"


def compile_codex_graph():
    """Compile the codex graph"""
    workflow = StateGraph(state_schema=CodexState, input=CodexInput, output=CodexOutput)
    
    # --- Add all nodes ---
    workflow.add_node("initialize-project", initialize_project)
    workflow.add_node("test-server-ready", test_server_ready) # Initial check
    workflow.add_node("planner", planner)
    
    # --- implementation loop nodes ---
    workflow.add_node("coder", coder)
    workflow.add_node("server-check", test_server_ready)
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("reflector", reflector)
    
    # --- Final step nodes ---
    workflow.add_node("raise-pr", raise_pr)
    workflow.add_node("deploy-service", deploy_service)
    workflow.add_node("finalize", finalize)

    # --- Wire the graph ---
    workflow.add_edge(START, "initialize-project")
    workflow.add_edge("initialize-project", "test-server-ready")

    # 1. Initial server check
    workflow.add_conditional_edges("test-server-ready", is_server_ready, {
        "planner": "planner",
        "end": END
    })

    # 2. Plan, then Implement
    workflow.add_edge("planner", "coder")

    # 3. After implement, always test
    workflow.add_edge("coder", "server-check")
    workflow.add_conditional_edges("server-check", is_codeer_implementation_working, {
        "evaluator": "evaluator",
        "coder": "coder",
    })

    # 4. After testing, decide what's next (THE LOOP)
    workflow.add_conditional_edges("evaluator", should_reflect_or_raise_pr, {
        "raise-pr": "raise-pr",
        "reflector": "reflector",
        "end": END
    })
    
    # 5. If reflector, loop back to planner
    workflow.add_edge("reflector", "planner")

    # 6. Final success path
    workflow.add_edge("raise-pr", "deploy-service")
    workflow.add_edge("deploy-service", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile(debug=True)

graph = compile_codex_graph()
