from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from agents.eval_agent.nodes.preflight import make_config_preflight_node, route_after_preflight
from agents.eval_agent.nodes.allignment.ensure_config import ensure_target_agent_config
from agents.eval_agent.nodes.allignment.generate_spec import generate_target_agent_spec

logger = get_logger("eval_agent.allignment")

def build_alignment_subgraph():
    """Build the agent spec subgraph."""
    builder = StateGraph(EvalAgentPlannerState)
    builder.add_node(
        "config-preflight",
        make_config_preflight_node(
            subgraph_name="alignment",
            required=["openai_api_key"],
        ),
    )
    builder.add_node("ensure-config", ensure_target_agent_config)
    builder.add_node("generate-agent-spec", generate_target_agent_spec)
    builder.add_edge(START, "config-preflight")
    builder.add_conditional_edges("config-preflight", route_after_preflight, {
        "continue": "ensure-config",
        "exit": END,
    })
    builder.add_edge("ensure-config", "generate-agent-spec")
    builder.add_edge("generate-agent-spec", END)
    return builder.compile()
