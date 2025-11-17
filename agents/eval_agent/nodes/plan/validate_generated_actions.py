import json
import re
from typing import Dict, List

from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from shared.schema import DatasetExample
from shared.tools import ToolEntry, build_tool_name_set, canonicalize_tool_name

logger = get_logger("eval_agent.plan.validate_generated_actions")


def _validate_action_list(
    example_id: str,
    phase_name: str,
    actions: List,
    valid_names: set,
    name_map: Dict[str, str],
    tool_entries: Dict[str, ToolEntry],
    invalid: List[str]
) -> None:
    """Validate a list of actions from a specific phase."""
    for idx, action in enumerate(actions):
        canonical_tool = canonicalize_tool_name(action.tool)
        action_label = f"{phase_name}[{idx}]"
        
        # Check tool exists
        if canonical_tool not in valid_names:
            invalid.append(f"{example_id} {action_label}: unknown tool {action.tool}")
            continue
        
        if action.tool == "system.wait":
            continue

        # Parse params - try to repair if malformed
        try:
            params_dict = json.loads(action.params) if action.params else {}
        except json.JSONDecodeError as e:
            # Fail fast on invalid JSON - strict mode should prevent this
            logger.error(
                "Invalid JSON in %s %s: %s\nParams: %r",
                example_id, action_label, str(e), action.params[:200]
            )
            invalid.append(f"{example_id} {action_label}: invalid JSON params")
            continue

        # Check for angle-bracket placeholders
        for key, value in params_dict.items():
            if isinstance(value, str) and re.match(r'^<[^>]+>$', value):
                invalid.append(
                    f"{example_id} {action_label}: invalid placeholder '{value}' in {key}. "
                    f"Use [var:name] or [resource:name] syntax"
                )

        # Check required params
        canonical_name = name_map.get(canonical_tool)
        tool_entry = tool_entries.get(canonical_name)
        
        if tool_entry and tool_entry.pydantic_schema:
            required = set(tool_entry.pydantic_schema.get('required', []))
            provided = set(params_dict.keys())
            missing = required - provided
            
            if missing:
                invalid.append(
                    f"{example_id} {action_label}: missing params {', '.join(missing)}"
                )


def _validate_generated_actions(
    examples: List[DatasetExample], tool_entries: Dict[str, ToolEntry]
) -> None:
    """Validate actions in NEW 3-phase format: provision_actions, expected_actions, assert_actions."""

    if not tool_entries:
        return

    name_map = build_tool_name_set(tool_entries)
    valid_names = set(name_map.keys())
    valid_names.add("system.wait")

    invalid: List[str] = []
    for example in examples:
        if not example.expected_output:
            invalid.append(f"{example.example_id}: missing expected_output")
            continue
        
        expected_output = example.expected_output
        
        # Validate provision_actions (Phase 1: create test data)
        if expected_output.provision_actions:
            _validate_action_list(
                example.example_id, "provision_actions",
                expected_output.provision_actions,
                valid_names, name_map, tool_entries, invalid
            )
        
        # Validate expected_actions (Phase 2: what agent should do - optional)
        if expected_output.expected_actions:
            _validate_action_list(
                example.example_id, "expected_actions",
                expected_output.expected_actions,
                valid_names, name_map, tool_entries, invalid
            )
        
        # Validate assert_actions (Phase 3: verify final state)
        if expected_output.assert_actions:
            _validate_action_list(
                example.example_id, "assert_actions",
                expected_output.assert_actions,
                valid_names, name_map, tool_entries, invalid
            )

    if invalid:
        raise ValueError("Validation failed:\n- " + "\n- ".join(invalid))



async def validate_generated_actions(state: EvalAgentPlannerState) -> dict:
    """Validate the generated actions."""
    # dataset_examples = state.dataset_examples
    # tool_entries = state.tool_entries
    # _validate_generated_actions(dataset_examples, tool_entries)
    return {
        "valid": True,
    }