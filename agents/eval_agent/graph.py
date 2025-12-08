"""overall graph for the eval agent"""
from typing import Literal
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
    workflow.add_node("langsmith_upload", _upload_run_results)
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
    workflow.add_edge("execute", "langsmith_upload")
    workflow.add_edge("langsmith_upload", "reflect")
    workflow.add_conditional_edges("reflect", should_continue, {
        "plan": "plan",
        "finalize": "finalize"
    })
    
    workflow.add_conditional_edges("finalize", should_start_new_round, {
        "update_state_from_handoff": "update_state_from_handoff",
        "__end__": END
    })
    workflow.add_edge("update_state_from_handoff", "plan")

    return workflow.compile(debug=True)


graph = build_graph()
