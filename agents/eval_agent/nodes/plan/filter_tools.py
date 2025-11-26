from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from shared.tools import ToolEntry
from typing import List, Dict

logger = get_logger("eval_agent.plan.filter_tools")


async def filter_tools(state: EvalAgentPlannerState) -> dict:
    """Filter the tools for the test generation."""
    tool_entries: Dict[str, ToolEntry] = {}
    context_for_scoring = ""
    for example in state.dataset_examples:
        context_for_scoring += ",".join(example.expected_output.create_test_data) + "\n" + ",".join(example.expected_output.assert_final_state)

    return {
        "tool_entries": tool_entries,
    }