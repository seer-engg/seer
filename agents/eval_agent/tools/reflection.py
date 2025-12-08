"""
Reflection phase tools for eval agent Supervisor pattern.
"""
import json
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

from shared.logger import get_logger
from shared.schema import AgentContext, ExperimentResultContext
from agents.eval_agent.models import EvalAgentState, Hypothesis
from agents.eval_agent.nodes.reflect.graph import reflect_node

logger = get_logger("eval_agent.tools.reflection")


@tool
async def reflect_on_results(runtime: ToolRuntime = None) -> str:
    """
    Analyze test results and generate hypothesis.
    
    Reads latest_results from state and generates reflection/hypothesis.
    
    Returns:
        JSON string with hypothesis
    """
    # Get state
    if not runtime or not hasattr(runtime, "state"):
        return json.dumps({"status": "error", "error": "No runtime state available"})
    
    state = runtime.state if isinstance(runtime.state, dict) else {}
    agent_context_dict = state.get("agent_context")
    latest_results_dicts = state.get("latest_results", [])
    dataset_examples_dicts = state.get("dataset_examples", [])
    
    if not latest_results_dicts:
        return json.dumps({"status": "error", "error": "No test results found"})
    
    try:
        # Reconstruct objects
        from shared.schema import DatasetExample
        agent_context = AgentContext(**agent_context_dict) if isinstance(agent_context_dict, dict) else agent_context_dict
        
        latest_results = []
        for r_dict in latest_results_dicts:
            if isinstance(r_dict, dict):
                latest_results.append(ExperimentResultContext(**r_dict))
            else:
                latest_results.append(r_dict)
        
        dataset_examples = []
        for ex_dict in dataset_examples_dicts:
            if isinstance(ex_dict, dict):
                dataset_examples.append(DatasetExample(**ex_dict))
            else:
                dataset_examples.append(ex_dict)
        
        # Create temporary state for node function
        temp_state = EvalAgentState(
            messages=[],
            context=agent_context,
            dataset_examples=dataset_examples,
            latest_results=latest_results,
            attempts=state.get("attempts", 0),
        )
        
        result = await reflect_node(temp_state)
        
        # Extract hypothesis from messages (reflect_node adds it to messages)
        hypothesis = None
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                # Try to extract hypothesis from message
                # The reflect_node should add a message with the hypothesis
                hypothesis_text = msg.content
                # For now, create a simple hypothesis
                hypothesis = Hypothesis(
                    summary=hypothesis_text[:500],
                    test_generation_critique=None
                )
                break
        
        if not hypothesis:
            # Fallback
            hypothesis = Hypothesis(
                summary="Reflection completed. Review test results for insights.",
                test_generation_critique=None
            )
        
        hypothesis_dict = hypothesis.model_dump() if hasattr(hypothesis, "model_dump") else hypothesis
        
        logger.info("✅ Reflection complete")
        return json.dumps({
            "status": "success",
            "hypothesis": hypothesis_dict,
            "message": "✅ Reflection complete: " + hypothesis.summary[:100]
        })
    except Exception as e:
        logger.error(f"Error reflecting on results: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)})

