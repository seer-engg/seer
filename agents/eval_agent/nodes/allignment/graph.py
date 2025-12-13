from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from agents.eval_agent.nodes.allignment.ensure_config import ensure_target_agent_config
from agents.eval_agent.nodes.allignment.generate_spec import generate_target_agent_spec
from agents.eval_agent.nodes.allignment.provision_target import provision_target_agent
logger = get_logger("eval_agent.allignment")

def build_alignment_subgraph():
    """Build the agent spec subgraph."""
    builder = StateGraph(EvalAgentPlannerState)
    builder.add_node("ensure-config", ensure_target_agent_config)
    builder.add_node("generate-agent-spec", generate_target_agent_spec)
    builder.add_node("provision-target", provision_target_agent)
    builder.add_edge(START, "ensure-config")
    builder.add_edge("ensure-config", "provision-target")
    builder.add_edge("provision-target", "generate-agent-spec")
    builder.add_edge("generate-agent-spec", END)
    return builder.compile()
