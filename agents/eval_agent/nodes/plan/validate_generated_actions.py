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
            
            if action.tool == "system.wait":
                continue  # 'system.wait' is a special case with no params

            # Now, let's get the tool's schema to check its parameters.
            # I'm assuming 'name_map' maps all aliases to the canonical name.
            canonical_name = name_map.get(canonicalize_tool_name(action.tool))
            
            tool_entry = tool_entries.get(canonical_name)
            
            # If there's no entry or it has no schema, we can't validate its params.
            # This follows the principle of Abstraction: if a tool doesn't publish
            # its 'contract' (its schema), we can't enforce it.
            if not tool_entry or not tool_entry.pydantic_schema:
                continue # Can't validate params for this tool, so we skip it

            schema = tool_entry.pydantic_schema
            
            # Find all required fields from the Pydantic model
            required_fields = set()
            # .model_fields is for Pydantic v2. Use .__fields__ if you're on v1
            for field_name, field in schema.model_fields.items():
                if field.is_required():
                    required_fields.add(field_name)

            # Find all fields the agent provided
            provided_fields = set(action.inputs.keys()) if action.inputs else set()

            # Now, check for any missing fields
            missing_fields = required_fields - provided_fields
            
            if missing_fields:
                invalid.append(
                    f"example={example.example_id or '<pending>'} action_index={idx} tool={action.tool} (Missing required params: {', '.join(missing_fields)})"
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