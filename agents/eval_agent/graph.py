"""overall graph for the eval agent"""
from typing import Literal, Optional
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.nodes.finalize import build_finalize_subgraph
from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.nodes.plan import build_plan_subgraph
from agents.eval_agent.nodes.reflect.graph import reflect_node
from agents.eval_agent.nodes.alignment import alignment_node
from agents.eval_agent.nodes.intent.classify import classify_user_intent
from agents.eval_agent.nodes.intent.answer import answer_informational_query
from shared.logger import get_logger
from agents.eval_agent.nodes.run import _prepare_run_context, _upload_run_results
from agents.eval_agent.nodes.execute import build_test_execution_subgraph
from shared.config import config

logger = get_logger("eval_agent.graph")

def update_state_from_handoff(state: EvalAgentState) -> dict:
    """
    Unpacks the codex_output handoff object into the main state for the next round.
    This is the single source of truth for updating state after a codex run.
    """
    codex_handoff = state.codex_output
    if not codex_handoff:
        raise ValueError("No codex handoff object found in state to update from.")

    logger.info("Updating state from codex handoff for the next evaluation round.")
    return {
        "target_agent_version": codex_handoff.target_agent_version,
        "sandbox_context": codex_handoff.updated_sandbox_context,
        "codex_output": None,  # Clear the handoff object after processing
    }


def should_continue(state: EvalAgentState) -> Literal["plan", "finalize"]:
    """Determine if the eval loop should continue plan or finalize."""
    return "plan" if state.attempts < config.eval_n_rounds else "finalize"


def should_execute_or_finalize(state: EvalAgentState) -> Literal["pre_run", "plan_summary"]:
    """After plan generation, decide: execute tests or return plan summary (plan-only mode)."""
    if config.eval_plan_only_mode:
        logger.info("📋 Plan-only mode enabled - skipping execution, returning plan summary")
        return "plan_summary"
    return "pre_run"


def should_show_alignment(state: EvalAgentState) -> Literal["alignment", "__end__"]:
    """After plan summary, check if alignment questions need to be shown."""
    if state.alignment_state and state.alignment_state.questions and not state.alignment_state.is_complete:
        logger.info("📋 Alignment questions present - routing to alignment node")
        return "alignment"
    logger.info("📋 No alignment questions or already complete - ending")
    return "__end__"


async def plan_summary_node(state: EvalAgentState) -> dict:
    """Return plan summary in plan-only mode. Generates AgentSpec and alignment questions."""
    from langchain_core.messages import AIMessage
    from agents.eval_agent.nodes.plan.generate_spec import generate_agent_spec_and_alignment
    
    num_tests = len(state.dataset_examples or [])
    logger.info("📋 Plan summary: %d test cases", num_tests)
    
    # Generate agent spec and alignment questions
    spec_data = await generate_agent_spec_and_alignment(state)
    
    # Format summary message
    agent_spec = spec_data["agent_spec"]
    alignment_state = spec_data["alignment_state"]
    
    summary_parts = [
        f"Plan generation complete: {num_tests} test cases generated.",
        f"\nAgent Specification:",
        f"- Name: {agent_spec.agent_name}",
        f"- Primary Goal: {agent_spec.primary_goal}",
        f"- Key Capabilities: {', '.join(agent_spec.key_capabilities[:3])}",
        f"- Confidence Score: {agent_spec.confidence_score:.2f}",
    ]
    
    if alignment_state.questions:
        summary_parts.append(f"\nAlignment Questions ({len(alignment_state.questions)}):")
        for i, q in enumerate(alignment_state.questions, 1):
            summary_parts.append(f"{i}. {q.question}")
            summary_parts.append(f"   Context: {q.context}")
    
    summary = "\n".join(summary_parts)
    
    return {
        "messages": [AIMessage(content=summary)],
        "agent_spec": spec_data["agent_spec"],
        "alignment_state": spec_data["alignment_state"],
    }


def should_start_new_round(state: EvalAgentState) -> Literal["update_state_from_handoff", "__end__"]:
    """
    Decision node based on the codex handoff object.
    If a valid handoff exists and we haven't hit the version limit, route to update state.
    """
    codex_handoff = state.codex_output
    if codex_handoff and codex_handoff.agent_updated and codex_handoff.target_agent_version < config.eval_n_versions:
        logger.info(f"Codex provided an update to v{codex_handoff.target_agent_version}. Starting new evaluation round.")
        return "update_state_from_handoff"
    else:
        if not codex_handoff or not codex_handoff.agent_updated:
            logger.info("Codex did not provide an update. Ending workflow.")
        else:
            logger.info(f"Reached max versions ({config.eval_n_versions}). Ending workflow.")
        return "__end__"


def route_by_intent(state: EvalAgentState) -> Literal["answer_informational", "plan"]:
    """Route based on user intent."""
    intent = state.user_intent
    if not intent:
        # Default to plan if intent not classified
        logger.warning("No user intent found, defaulting to plan")
        return "plan"
    
    if intent.intent_type == "informational":
        logger.info(f"📋 Informational query detected (confidence: {intent.confidence:.2f}) - answering directly")
        return "answer_informational"
    else:
        logger.info(f"📋 Evaluation request detected (confidence: {intent.confidence:.2f}) - proceeding to plan")
        return "plan"


def build_graph():
    """Build the evaluation agent graph."""
    workflow = StateGraph(EvalAgentState)
    plan_subgraph = build_plan_subgraph()
    finalize_subgraph = build_finalize_subgraph()

    # Intent classification nodes
    workflow.add_node("classify_intent", classify_user_intent)
    workflow.add_node("answer_informational", answer_informational_query)
    
    # Existing nodes
    workflow.add_node("plan", plan_subgraph)
    workflow.add_node("plan_summary", plan_summary_node)
    workflow.add_node("alignment", alignment_node)
    workflow.add_node("pre_run", _prepare_run_context)
    workflow.add_node("execute", build_test_execution_subgraph())
    workflow.add_node("langfuse_upload", _upload_run_results)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("finalize", finalize_subgraph)
    workflow.add_node("update_state_from_handoff", update_state_from_handoff)

    # Start with intent classification, then route
    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges("classify_intent", route_by_intent, {
        "answer_informational": "answer_informational",
        "plan": "plan"
    })
    workflow.add_edge("answer_informational", END)
    # Conditional: plan-only mode skips execution
    workflow.add_conditional_edges("plan", should_execute_or_finalize, {
        "pre_run": "pre_run",
        "plan_summary": "plan_summary"
    })
    # Conditional: after plan_summary, check if alignment needed
    workflow.add_conditional_edges("plan_summary", should_show_alignment, {
        "alignment": "alignment",
        "__end__": END
    })
    # After alignment, end (user can continue conversation to refine further)
    workflow.add_edge("alignment", END)
    workflow.add_edge("pre_run", "execute")
    workflow.add_edge("execute", "langfuse_upload")
    workflow.add_edge("langfuse_upload", "reflect")
    workflow.add_conditional_edges("reflect", should_continue, {
        "plan": "plan",
        "finalize": "finalize"
    })
    
    workflow.add_conditional_edges("finalize", should_start_new_round, {
        "update_state_from_handoff": "update_state_from_handoff",
        "__end__": END
    })
    workflow.add_edge("update_state_from_handoff", "plan")

    # Initialize checkpointer for human-in-the-loop interrupts
    checkpointer = None
    if config.database_uri:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            logger.info(f"Initializing PostgresSaver checkpointer with database URI")
            checkpointer = PostgresSaver.from_conn_string(config.database_uri)
            # Setup tables on first run (idempotent)
            try:
                checkpointer.setup()
                logger.info("PostgresSaver checkpointer setup complete")
            except Exception as e:
                # Tables might already exist, which is fine
                logger.debug(f"PostgresSaver setup (tables may already exist): {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgresSaver checkpointer: {e}. Interrupts will not work.")
            logger.warning("Set DATABASE_URI environment variable to enable human-in-the-loop interrupts.")
    else:
        logger.warning("DATABASE_URI not set. Human-in-the-loop interrupts will not work.")
        logger.warning("Set DATABASE_URI environment variable to enable interrupts.")

    compiled_graph = workflow.compile(debug=True, checkpointer=checkpointer)
    
    # Configure Langfuse callbacks at graph compilation time for LangGraph dev server
    # This ensures traces are created even when graph is invoked via HTTP
    # CRITICAL: For LangGraph dev server, callbacks MUST be configured at compile time
    # using .with_config(), not just passed when invoking via RemoteGraph
    # See: https://langfuse.com/guides/cookbook/integration_langgraph
    from agents.eval_agent.constants import LANGFUSE_CLIENT
    if LANGFUSE_CLIENT and config.langfuse_public_key:
        try:
            from langfuse.langchain import CallbackHandler
            from langfuse import propagate_attributes
            
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
                metadata={"project_name": config.project_name}
            )
            
            # Use with_config to attach callbacks at compile time
            # This is required for LangGraph dev server to capture traces
            compiled_graph = compiled_graph.with_config({
                "callbacks": [langfuse_handler]
            })
            
            logger.info(f"📊 Langfuse tracing configured at graph compilation time with project_name={config.project_name}")
        except Exception as e:
            logger.warning(f"Failed to configure Langfuse callbacks at graph compilation: {e}")
    
    return compiled_graph


graph = build_graph()
