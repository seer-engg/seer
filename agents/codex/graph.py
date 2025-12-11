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
    developer, evaluator, reflector, finalize,
    initialize_project, index
)
from shared.config import config
import os

logger = get_logger("codex.graph")


def is_server_ready(state: CodexState) -> CodexState:
    """This router checks if the server is ready at the *start*"""
    if state.server_running:
        return "index-codebase"
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
        if config.allow_pr:            
            return "raise-pr"
        else:
            return "end"
        
    logger.info(f"Implementation failed (Attempt {state.attempt_number}). Reflecting.")
    return "reflector"

def is_codeer_implementation_working(state: CodexState) -> CodexState:
    """This router checks if the codeer implementation is working"""
    if state.server_running:
        return "evaluator"
    else:
        return "developer"


def compile_codex_graph():
    """Compile the codex graph"""
    workflow = StateGraph(state_schema=CodexState, input=CodexInput, output=CodexOutput)
    
    # --- Add all nodes ---
    workflow.add_node("initialize-project", initialize_project)
    workflow.add_node("test-server-ready", test_server_ready) # Initial check
    workflow.add_node("index-codebase", index)
    
    # --- implementation loop nodes ---
    workflow.add_node("developer", developer)
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
        "index-codebase": "index-codebase",
        "end": END
    })
    workflow.add_edge("index-codebase", "reflector")
    workflow.add_edge("reflector", "developer")

    # 3. After implement, always test
    workflow.add_edge("developer", "server-check")
    workflow.add_conditional_edges("server-check", is_codeer_implementation_working, {
        "evaluator": "evaluator",
        "developer": "developer",
    })

    # 4. After testing, decide what's next (THE LOOP)
    workflow.add_conditional_edges("evaluator", should_reflect_or_raise_pr, {
        "raise-pr": "raise-pr",
        "reflector": "reflector",
        "end": END
    })
    
    # 6. Final success path
    workflow.add_edge("raise-pr", "deploy-service")
    workflow.add_edge("deploy-service", "finalize")
    workflow.add_edge("finalize", END)

    compiled_graph = workflow.compile(debug=True)
    
    # Configure Langfuse callbacks at graph compilation time for LangGraph dev server
    # This ensures traces are created even when graph is invoked via HTTP
    # CRITICAL: For LangGraph dev server, callbacks MUST be configured at compile time
    # using .with_config(), not just passed when invoking via RemoteGraph
    # See: https://langfuse.com/guides/cookbook/integration_langgraph
    from agents.eval_agent.constants import LANGFUSE_CLIENT
    if LANGFUSE_CLIENT and config.langfuse_public_key:
        try:
            from langfuse.langchain import CallbackHandler
            
            # Create a custom wrapper that adds metadata to the root trace
            # CRITICAL: For LangGraph dev server, metadata must be set when the root chain starts
            class MetadataCallbackHandler(CallbackHandler):
                def __init__(self, *args, metadata=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.metadata = metadata or {}
                    self._root_chain_started = False
                
                def on_chain_start(self, serialized, inputs, **kwargs):
                    """Override to add metadata when LangGraph root chain starts"""
                    result = super().on_chain_start(serialized, inputs, **kwargs)
                    
                    # Detect LangGraph root chain start (name is "LangGraph" or id is None)
                    run_id = kwargs.get("run_id")
                    parent_run_id = kwargs.get("parent_run_id")
                    
                    # Root chain has no parent_run_id and is the LangGraph chain
                    if not self._root_chain_started and parent_run_id is None:
                        self._root_chain_started = True
                        from langfuse import get_client, propagate_attributes
                        langfuse = get_client()
                        try:
                            # Use propagate_attributes to ensure metadata is attached to root trace
                            # This is the same approach Supervisor uses
                            with propagate_attributes(metadata=self.metadata):
                                # The metadata will now be propagated to all observations
                                # Update the trace directly as well for immediate effect
                                langfuse.update_current_trace(metadata=self.metadata)
                                logger.debug(f"✅ Added metadata to root trace: {self.metadata}")
                        except Exception as e:
                            logger.warning(f"Failed to add metadata to root trace: {e}")
                    
                    return result
            
            langfuse_handler = MetadataCallbackHandler(
                public_key=config.langfuse_public_key,
                metadata={"project_name": config.codex_project_name}
            )
            
            # Use with_config to attach callbacks at compile time
            # This is required for LangGraph dev server to capture traces
            compiled_graph = compiled_graph.with_config({
                "callbacks": [langfuse_handler]
            })
            
            logger.info(f"📊 Langfuse tracing configured at graph compilation time with project_name={config.codex_project_name}")
        except Exception as e:
            logger.warning(f"Failed to configure Langfuse callbacks at graph compilation: {e}")
    
    return compiled_graph

graph = compile_codex_graph()
