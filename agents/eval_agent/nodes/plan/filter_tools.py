from typing import Dict, List
from agents.eval_agent.models import EvalAgentPlannerState
from agents.eval_agent.nodes.execute.utils import get_tool_hub
from shared.logger import get_logger
from shared.tools import ToolEntry

logger = get_logger("eval_agent.plan.filter_tools")


async def filter_tools(state: EvalAgentPlannerState) -> dict:
    """
    Filter tools using semantic search against the generated test plan.
    This reduces the context window and noise for the execution agents.
    """
    
    # 1. Get the Hub (reusing the cached instance logic from utils)
    # Note: get_tool_hub accepts state for context but doesn't strictly require TestExecutionState fields
    # We cast to any to avoid type checker complaints if strict checking were enabled
    hub = await get_tool_hub(state)  # type: ignore
    
    # 2. Collect all instructions that need tool support from all examples
    all_instructions = []
    for example in state.dataset_examples:
        if example.expected_output:
            # Combine phase 1 (provision) and phase 3 (assert) instructions
            # The planner generates these as lists of strings
            if example.expected_output.create_test_data:
                all_instructions.extend(example.expected_output.create_test_data)
            if example.expected_output.assert_final_state:
                all_instructions.extend(example.expected_output.assert_final_state)

    query_text = "\n".join(all_instructions)
    if not query_text:
        logger.warning("No instructions found to filter tools against.")
        return {"tool_entries": {}}

    # 3. Query the Hub
    # We ask for a generous number (e.g. 20) to cover multiple complex steps across examples
    logger.info(f"Filtering tools for instruction set length: {len(query_text)}")
    
    # ToolHub.query performs semantic search + graph expansion (finding dependent tools)
    # It returns a list of dictionaries compatible with OpenAI tool schema
    # Wrapping in to_thread because hub.query performs blocking network calls (embeddings)
    import asyncio
    relevant_tool_dicts = await asyncio.to_thread(hub.query, query_text, top_k=20)
    
    # 4. Convert to ToolEntry format expected by the agent state
    tool_entries: Dict[str, ToolEntry] = {}
    for t_dict in relevant_tool_dicts:
        name = t_dict.get("name")
        if not name:
            continue
            
        # Infer service from name (e.g. 'github_create_issue' -> 'github')
        # This is a heuristic; ideally ToolHub would return this metadata
        service = name.split("_")[0] if "_" in name else "general"
        
        tool_entries[name] = ToolEntry(
            name=name,
            description=t_dict.get("description", ""),
            service=service,
            pydantic_schema=t_dict.get("parameters") # Maps JSON schema to pydantic_schema field
        )

    logger.info(f"Selected {len(tool_entries)} tools: {list(tool_entries.keys())}")

    return {
        "tool_entries": tool_entries,
    }
