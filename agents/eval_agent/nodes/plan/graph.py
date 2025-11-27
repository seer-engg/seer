"""
This file contains code for the plan node of the eval agent. 
This is also responsible for generating the test cases for the target agent.
"""
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from agents.eval_agent.nodes.plan.ensure_config import ensure_target_agent_config
from agents.eval_agent.nodes.plan.provision_target import provision_target_agent
from agents.eval_agent.nodes.plan.get_reflections import get_reflections
from agents.eval_agent.nodes.plan.agentic_eval_generation import agentic_eval_generation
from agents.eval_agent.nodes.plan.validate_generated_actions import validate_generated_actions
from agents.eval_agent.nodes.plan.filter_tools import filter_tools

logger = get_logger("eval_agent.plan")

def build_plan_subgraph():
    """Build the plan subgraph."""
    builder = StateGraph(EvalAgentPlannerState)
    builder.add_node("ensure-config", ensure_target_agent_config)
    builder.add_node("provision-target", provision_target_agent)
    builder.add_node("get-reflections", get_reflections)
    builder.add_node("eval-gen", agentic_eval_generation)
    builder.add_node("validate-generated-actions", validate_generated_actions)
    builder.add_node("filter-tools", filter_tools)
    builder.add_edge(START, "ensure-config")
    builder.add_edge("ensure-config", "provision-target")
    builder.add_edge("provision-target", "get-reflections")
    builder.add_edge("get-reflections", "eval-gen")
    builder.add_edge("eval-gen", "filter-tools")
    builder.add_edge("filter-tools", "validate-generated-actions")
    builder.add_edge("validate-generated-actions", END)

    return builder.compile()
