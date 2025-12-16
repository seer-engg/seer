from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from agents.eval_agent.nodes.testing.run import prepare_run_context, upload_run_results
from agents.eval_run import build_test_execution_subgraph
logger = get_logger("eval_agent.testing.graph")

def build_testing_subgraph():
    """Build the testing subgraph."""
    builder = StateGraph(EvalAgentState)

    execute_subgraph = build_test_execution_subgraph()
    builder.add_node("prepare_run_context", prepare_run_context)
    builder.add_node("upload_run_results", upload_run_results)
    builder.add_node("execute", execute_subgraph)
    builder.add_edge(START, "prepare_run_context")
    builder.add_edge("prepare_run_context", "execute")
    builder.add_edge("execute", "upload_run_results")
    builder.add_edge("upload_run_results", END)
    return builder.compile()