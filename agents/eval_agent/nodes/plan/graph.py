"""
This file contains code for the plan node of the eval agent. 
This is also responsible for generating the test cases for the target agent.
"""
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from agents.eval_agent.nodes.plan.ensure_config import ensure_target_agent_config
from agents.eval_agent.nodes.plan.provision_target import provision_target_agent
from agents.eval_agent.nodes.plan.generate_evals import generate_eval_plan
from agents.eval_agent.nodes.plan.configure_target import configure_target_agent

logger = get_logger("eval_agent.plan")

def build_plan_subgraph():
    """Build the plan subgraph."""
    builder = StateGraph(EvalAgentState)
    builder.add_node("ensure-config", ensure_target_agent_config)
    builder.add_node("provision-target", provision_target_agent)
    builder.add_node("configure-target", configure_target_agent)
    builder.add_node("generate-tests", generate_eval_plan)

    builder.add_edge(START, "ensure-config")
    builder.add_edge("ensure-config", "provision-target")
    builder.add_edge("provision-target", "configure-target")
    builder.add_edge("configure-target", "generate-tests")
    builder.add_edge("generate-tests", END)

    return builder.compile()
