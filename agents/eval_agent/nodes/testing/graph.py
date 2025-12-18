from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from agents.eval_agent.nodes.preflight import make_config_preflight_node, route_after_preflight
from agents.eval_agent.nodes.testing.run import prepare_run_context, upload_run_results
from agents.eval_run import build_test_execution_subgraph
logger = get_logger("eval_agent.testing.graph")
from agents.eval_agent.nodes.testing.provision_target import provision_target_agent

def build_testing_subgraph():
    """Build the testing subgraph."""
    builder = StateGraph(EvalAgentState)

    execute_subgraph = build_test_execution_subgraph()
    builder.add_node(
        "config-preflight",
        make_config_preflight_node(
            subgraph_name="testing",
            required=lambda state: (
                ["openai_api_key", "github_token"]
                + (["composio_api_key"] if getattr(state.context, "mcp_services", []) else [])
            ),
        ),
    )
    builder.add_node("prepare_run_context", prepare_run_context)
    builder.add_node("upload_run_results", upload_run_results)
    builder.add_node("provision-target", provision_target_agent)
    builder.add_node("execute", execute_subgraph)
    builder.add_edge(START, "config-preflight")
    builder.add_conditional_edges("config-preflight", route_after_preflight, {
        "continue": "prepare_run_context",
        "exit": END,
    })
    builder.add_edge("prepare_run_context", "provision-target")
    builder.add_edge("provision-target", "execute")
    builder.add_edge("execute", "upload_run_results")
    builder.add_edge("upload_run_results", END)
    return builder.compile()