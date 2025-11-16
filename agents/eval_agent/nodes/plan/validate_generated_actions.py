import json
from typing import Dict, List

from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from shared.schema import DatasetExample
from shared.tools import ToolEntry, build_tool_name_set, canonicalize_tool_name

logger = get_logger("eval_agent.plan.validate_generated_actions")


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
            # Canonicalize the tool name for validation
            canonical_tool = canonicalize_tool_name(action.tool)
            if canonical_tool not in valid_names:
                invalid.append(
                    f"example={example.example_id or '<pending>'} action_index={idx} tool={action.tool}"
                )
            
            if action.tool == "system.wait":
                continue  # 'system.wait' is a special case with no params

            # Now, let's get the tool's schema to check its parameters.
            # I'm assuming 'name_map' maps all aliases to the canonical name.
            canonical_name = name_map.get(canonical_tool)
            
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

            # Parse the params JSON string to get provided fields
            try:
                params_dict = json.loads(action.params) if action.params else {}
                provided_fields = set(params_dict.keys())
            except json.JSONDecodeError:
                # If params is not valid JSON, consider it as having no fields
                provided_fields = set()

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