"""
This file contains code for the plan node of the eval agent. 
This is also responsible for generating the test cases for the target agent.
"""
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from agents.eval_agent.nodes.preflight import make_config_preflight_node, route_after_preflight
from agents.eval_agent.nodes.plan.get_reflections import get_reflections
from agents.eval_agent.nodes.plan.agentic_eval_generation import agentic_eval_generation
from agents.eval_agent.nodes.plan.validate_generated_actions import validate_generated_actions
from agents.eval_agent.nodes.plan.filter_tools import filter_tools
from langchain_core.messages import AIMessage

async def generate_response(state: EvalAgentPlannerState) -> dict:
    """Generate a response for the target agent."""
    dataset_examples_markdown = "\n".join([example.to_markdown() for example in state.dataset_examples])
    output_messages = [AIMessage(content=f"### Test Cases\n{dataset_examples_markdown}")]
    return {
        "messages": output_messages,
    }

logger = get_logger("eval_agent.plan")

def build_plan_subgraph():
    """Build the plan subgraph."""
    builder = StateGraph(EvalAgentPlannerState)
    builder.add_node(
        "config-preflight",
        make_config_preflight_node(
            subgraph_name="plan",
            required=["openai_api_key"],
        ),
    )
    builder.add_node("get-reflections", get_reflections)
    builder.add_node("eval-gen", agentic_eval_generation)
    builder.add_node("validate-generated-actions", validate_generated_actions)
    builder.add_node("filter-tools", filter_tools)
    builder.add_node("generate-response", generate_response)
    builder.add_edge(START, "config-preflight")
    builder.add_conditional_edges("config-preflight", route_after_preflight, {
        "continue": "get-reflections",
        "exit": END,
    })
    builder.add_edge("get-reflections", "eval-gen")
    builder.add_edge("eval-gen", "filter-tools")
    builder.add_edge("filter-tools", "validate-generated-actions")
    builder.add_edge("validate-generated-actions", "generate-response")
    builder.add_edge("generate-response", END)

    return builder.compile()
