from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
logger = get_logger("eval_agent.plan.validate_generated_actions")
from typing import Dict, List
from shared.schema import DatasetExample
from shared.tool_catalog import ToolEntry, build_tool_name_set, canonicalize_tool_name


def _validate_generated_actions(
    examples: List[DatasetExample], tool_entries: Dict[str, ToolEntry]
) -> None:
    """Ensure every generated action references a known tool."""

    if not tool_entries:
        return

    name_map = build_tool_name_set(tool_entries)
    valid_names = set(name_map.keys())
    valid_names.add("system.wait")

    invalid: List[str] = []
    for example in examples:
        # ADDED: Handle cases where expected_output might be None
        if not example.expected_output:
            invalid.append(
                f"example={example.example_id or '<pending>'} has missing expected_output"
            )
            continue
        for idx, action in enumerate(example.expected_output.actions):
            if action.tool not in valid_names:
                invalid.append(
                    f"example={example.example_id or '<pending>'} action_index={idx} tool={action.tool}"
                )

    if invalid:
        sample = ", ".join(list(name_map.values())[:20])
        raise ValueError(
            "Generated actions referenced unknown tools or had missing data. Details: "
            + ", ".join(invalid)
            + (f". Known tools include: {sample}" if sample else "")
        )



async def validate_generated_actions(state: EvalAgentPlannerState) -> dict:
    """Validate the generated actions."""
    dataset_examples = state.dataset_examples
    tool_entries = state.tool_entries
    _validate_generated_actions(dataset_examples, tool_entries)
    return {
        "valid": True,
    }