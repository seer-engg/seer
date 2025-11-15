"""
This file contains code for the plan node of the eval agent. 
This is also responsible for generating the test cases for the target agent.
"""
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from agents.eval_agent.nodes.plan.ensure_config import ensure_target_agent_config
from agents.eval_agent.nodes.plan.provision_target import provision_target_agent
from agents.eval_agent.nodes.plan.configure_target import configure_target_agent
from agents.eval_agent.nodes.plan.get_reflections_and_tools import get_reflections_and_tools
from agents.eval_agent.nodes.plan.genetic_eval_generation import genetic_eval_generation
from agents.eval_agent.nodes.plan.agentic_eval_generation import agentic_eval_generation
from agents.eval_agent.nodes.plan.validate_generated_actions import validate_generated_actions

logger = get_logger("eval_agent.plan")

async def eval_generation(state: EvalAgentPlannerState) -> dict:
    if state.use_genetic_test_generation:
        return "genetic-eval-generation"
    else:
        return "agentic-eval-generation"

def build_plan_subgraph():
    """Build the plan subgraph."""
    builder = StateGraph(EvalAgentPlannerState)
    builder.add_node("ensure-config", ensure_target_agent_config)
    builder.add_node("provision-target", provision_target_agent)
    builder.add_node("configure-target", configure_target_agent)
    builder.add_node("get-reflections-and-tools", get_reflections_and_tools)
    builder.add_node("genetic-eval-generation", genetic_eval_generation)
    builder.add_node("agentic-eval-generation", agentic_eval_generation)
    builder.add_node("validate-generated-actions", validate_generated_actions)
    builder.add_edge(START, "ensure-config")
    builder.add_edge("ensure-config", "provision-target")
    builder.add_edge("provision-target", "configure-target")
    builder.add_edge("configure-target", "get-reflections-and-tools")
    builder.add_conditional_edges("get-reflections-and-tools", eval_generation, {
        "genetic-eval-generation": "genetic-eval-generation",
        "agentic-eval-generation": "agentic-eval-generation",
    })
    builder.add_edge("genetic-eval-generation", "validate-generated-actions")
    builder.add_edge("agentic-eval-generation", "validate-generated-actions")
    builder.add_edge("validate-generated-actions", END)

    return builder.compile()
