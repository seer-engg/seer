"""
Finalization phase tools for eval agent Supervisor pattern.
"""
import json
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

from shared.logger import get_logger
from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.nodes.finalize import build_finalize_subgraph

logger = get_logger("eval_agent.tools.finalization")


@tool
async def finalize_evaluation(runtime: ToolRuntime = None) -> str:
    """
    Finalize evaluation: handoff to codex (if enabled) and summarize.
    
    Reads all state and generates final summary.
    
    Returns:
        JSON string with final summary
    """
    # Get state
    if not runtime or not hasattr(runtime, "state"):
        return json.dumps({"status": "error", "error": "No runtime state available"})
    
    state = runtime.state if isinstance(runtime.state, dict) else {}
    
    try:
        # Reconstruct full state for finalize nodes
        from shared.schema import AgentContext, DatasetExample, ExperimentResultContext, AgentSpec
        
        agent_context = None
        if state.get("agent_context"):
            agent_context_dict = state.get("agent_context")
            agent_context = AgentContext(**agent_context_dict) if isinstance(agent_context_dict, dict) else agent_context_dict
        
        dataset_examples = []
        for ex_dict in state.get("dataset_examples", []):
            if isinstance(ex_dict, dict):
                dataset_examples.append(DatasetExample(**ex_dict))
            else:
                dataset_examples.append(ex_dict)
        
        latest_results = []
        for r_dict in state.get("latest_results", []):
            if isinstance(r_dict, dict):
                latest_results.append(ExperimentResultContext(**r_dict))
            else:
                latest_results.append(r_dict)
        
        temp_state = EvalAgentState(
            messages=[],
            context=agent_context or AgentContext(),
            dataset_examples=dataset_examples,
            latest_results=latest_results,
            attempts=state.get("attempts", 0),
            agent_spec=None,  # Will be set if available
        )
        
        # Use finalize subgraph
        finalize_subgraph = build_finalize_subgraph()
        config = {"recursion_limit": 50}
        result = await finalize_subgraph.ainvoke(temp_state, config)
        
        # Extract final message
        messages = result.get("messages", [])
        final_summary = ""
        if messages:
            last_msg = messages[-1]
            final_summary = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        
        logger.info("✅ Evaluation finalized")
        return json.dumps({
            "status": "success",
            "summary": final_summary,
            "message": "✅ Evaluation finalized"
        })
    except Exception as e:
        logger.error(f"Error finalizing evaluation: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)})

