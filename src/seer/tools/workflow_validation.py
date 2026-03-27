"""
Shared workflow validation utilities.

Used by both Nexus agent tools and MCP tools to validate workflow specs.
Provides consistent validation logic for tool references, trigger references,
and full compilation validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from seer.logger import get_logger

logger = get_logger(__name__)


class ValidationError:
    """Structured validation error with optional hint."""

    def __init__(self, error_type: str, message: str, hint: Optional[str] = None):
        self.error_type = error_type
        self.message = message
        self.hint = hint

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"error_type": self.error_type, "message": self.message}
        if self.hint:
            result["hint"] = self.hint
        return result


@dataclass
class ValidationResult:
    """Result of the full workflow validation pipeline."""

    success: bool
    """Whether the entire validation pipeline passed."""

    validated_spec: Optional[Any] = None
    """The Pydantic-parsed WorkflowSpec (post-trigger-fix). None on failure."""

    fixed_spec_dict: Optional[Dict[str, Any]] = None
    """The spec dict after trigger auto-fixes were applied. None on failure."""

    error: Optional[ValidationError] = None
    """Structured error if validation failed at any stage. None on success."""

    schema_fixes: List[Dict[str, Any]] = field(default_factory=list)
    """Records of trigger event_schema auto-fixes applied (may be empty)."""


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """
    Get attribute or dict key from object.

    Allows validation functions to work with both dict and Pydantic model inputs.
    """
    if hasattr(obj, name) and not isinstance(obj, dict):
        return getattr(obj, name, default)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def validate_tool_references(spec: Any) -> List[str]:
    """
    Check that all tool nodes reference valid tools in the registry.

    Args:
        spec: Workflow spec (dict or Pydantic model)

    Returns:
        List of error messages for invalid tool references
    """
    # Import here to avoid circular imports
    from seer.tools.base import get_tool  # pylint: disable=import-outside-toplevel

    errors = []
    nodes = _get_attr_or_key(spec, "nodes", [])

    for node in nodes:
        node_type = _get_attr_or_key(node, "type")
        if node_type == "tool":
            tool_name = _get_attr_or_key(node, "tool")
            if tool_name and not get_tool(tool_name):
                # Extract integration prefix for helpful hint
                prefix = tool_name.split('_')[0] if '_' in tool_name else tool_name
                errors.append(
                    f"Tool '{tool_name}' not found. "
                    f"Use search_tools('{prefix}') to find the correct tool name."
                )

    return errors


def validate_trigger_references(spec: Any) -> List[str]:
    """
    Check that all triggers reference valid trigger keys in the registry.

    Args:
        spec: Workflow spec (dict or Pydantic model)

    Returns:
        List of error messages for invalid trigger references
    """
    # Import here to avoid circular imports
    from seer.core.registry.trigger_registry import trigger_registry  # pylint: disable=import-outside-toplevel

    errors = []
    triggers = _get_attr_or_key(spec, "triggers", [])

    if triggers:
        available_triggers = [t.key for t in trigger_registry.all()]
        for trigger in triggers:
            trigger_key = _get_attr_or_key(trigger, "key")
            if trigger_key and not trigger_registry.maybe_get(trigger_key):
                errors.append(
                    f"Trigger '{trigger_key}' not found. "
                    f"Available triggers: {', '.join(available_triggers)}. "
                    f"Use search_triggers() to find the correct trigger key."
                )

    return errors


def validate_trigger_provider_configs(spec: Any) -> List[str]:
    """
    Validate provider_config for each trigger against its config schema.

    Args:
        spec: Workflow spec (dict or Pydantic model)

    Returns:
        List of error messages for invalid provider_config values
    """
    # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module level
    from seer.core.registry.trigger_registry import trigger_registry
    from jsonschema import Draft7Validator

    errors = []
    triggers = _get_attr_or_key(spec, "triggers", [])

    for trigger in triggers:
        trigger_key = _get_attr_or_key(trigger, "key")
        provider_config = _get_attr_or_key(trigger, "provider_config", {})

        if not trigger_key or not provider_config:
            continue

        definition = trigger_registry.maybe_get(trigger_key)
        if not definition or not definition.schemas.config:
            continue

        validator = Draft7Validator(definition.schemas.config)
        validation_errors = list(validator.iter_errors(provider_config))

        if validation_errors:
            trigger_id = _get_attr_or_key(trigger, "id", trigger_key)
            schema_props = list(definition.schemas.config.get("properties", {}).keys()) if isinstance(definition.schemas.config, dict) else []
            for err in validation_errors:
                hint = f" Expected fields: {schema_props}" if schema_props else ""
                errors.append(
                    f"Trigger '{trigger_id}' ({trigger_key}): {err.message}.{hint}"
                )

    return errors


def validate_tools_and_triggers(spec: Any) -> List[str]:
    """
    Validate both tool and trigger references.

    Convenience function that combines tool and trigger validation.

    Args:
        spec: Workflow spec (dict or Pydantic model)

    Returns:
        Combined list of error messages
    """
    return validate_tool_references(spec) + validate_trigger_references(spec)


def _build_reference_hint(spec: Dict[str, Any], exc: Exception) -> str:
    """Build an actionable hint for ValidationPhaseError by looking up node output schemas."""
    import re  # pylint: disable=import-outside-toplevel # Reason: Only needed for error parsing
    from seer.tools.base import get_tool  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    base_hint = "Check that all ${...} references point to valid variables."
    error_msg = str(exc)

    # Try to extract node_id from patterns like "node_id.property" or "${node_id.property}"
    match = re.search(r"(?:\$\{)?(\w+)\.(\w+)", error_msg)
    if not match:
        return base_hint

    node_id = match.group(1)
    nodes = spec.get("nodes", [])

    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("id") != node_id:
            continue
        tool_name = node.get("tool")
        if not tool_name:
            break
        tool_def = get_tool(tool_name)
        if not tool_def:
            break
        output_schema = getattr(tool_def, "output_schema", None) or {}
        if isinstance(output_schema, dict):
            props = sorted(output_schema.get("properties", {}).keys())
            if props:
                return (
                    f"Node '{node_id}' (tool: {tool_name}) outputs: {props}. "
                    f"Use one of these property names in your ${{...}} references."
                )
        break

    return base_hint


async def validate_compilation(
    user: Any,
    spec: Dict[str, Any],
    *,
    detailed_errors: bool = True
) -> Optional[ValidationError]:
    """
    Run full compilation validation.

    Validates type environment, references, and full workflow compilation.

    Args:
        user: User object for compilation context
        spec: Workflow spec as dict
        detailed_errors: If True, return structured errors with hints.
                        If False, return simpler error messages.

    Returns:
        ValidationError if compilation fails, None if successful
    """
    # Import here to avoid circular imports
    from seer.core.runtime.global_compiler import WorkflowCompilerSingleton  # pylint: disable=import-outside-toplevel
    from seer.core.errors import (  # pylint: disable=import-outside-toplevel
        ValidationPhaseError,
        TypeEnvironmentError,
        WorkflowCompilerError,
    )

    try:
        compiler = WorkflowCompilerSingleton.instance()
        await compiler.compile(user, spec, checkpointer=None)
        return None
    except TypeEnvironmentError as exc:
        logger.warning("Workflow type environment validation failed", exc_info=exc)
        error_type = "type_environment" if detailed_errors else "compilation"
        message = f"Type validation failed: {exc}" if detailed_errors else str(exc)
        hint = (
            "Check that output schemas match input expectations. "
            "Common issue: field name mismatches like 'threadId' vs 'thread_id'."
        ) if detailed_errors else None
        return ValidationError(error_type, message, hint)
    except ValidationPhaseError as exc:
        logger.warning("Workflow reference validation failed", exc_info=exc)
        error_type = "validation" if detailed_errors else "compilation"
        message = f"Validation failed: {exc}" if detailed_errors else str(exc)
        hint = _build_reference_hint(spec, exc) if detailed_errors else None
        return ValidationError(error_type, message, hint)
    except WorkflowCompilerError as exc:
        logger.warning("Workflow compilation failed", exc_info=exc)
        return ValidationError("compilation", f"Compilation failed: {exc}")
    except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Catch unexpected compilation errors gracefully
        logger.warning("Unexpected compilation error", exc_info=exc)
        return ValidationError("compilation", str(exc))


def detect_extra_fields(spec_dict: Dict[str, Any], error_msg: str) -> Optional[str]:
    """
    Detect extra fields in spec and return helpful hint.

    Analyzes Pydantic validation errors to provide user-friendly hints
    about invalid fields in the workflow spec.

    Args:
        spec_dict: The workflow spec dict
        error_msg: The Pydantic validation error message

    Returns:
        Hint string if extra fields detected, None otherwise
    """
    if "extra_forbidden" not in error_msg and "extra inputs" not in error_msg.lower():
        return None

    valid_fields = {"version", "nodes", "edges", "triggers"}
    invalid_fields = [k for k in spec_dict.keys() if k not in valid_fields]

    if not invalid_fields:
        return None

    return (
        "WorkflowSpec v2 schema ONLY allows: version, nodes, edges, triggers. "
        f"Invalid fields: {invalid_fields}. "
        "Remove: input_variables, inputs, config, metadata, or custom fields. "
        "Access trigger data via: ${trigger.data.field_name}"
    )


def format_validation_errors(errors: List[str], error_type: str = "reference_validation") -> ValidationError:
    """
    Format a list of validation errors into a ValidationError.

    Args:
        errors: List of error message strings
        error_type: The error type category

    Returns:
        ValidationError with combined message and hint
    """
    return ValidationError(
        error_type,
        "Workflow references non-existent tools or triggers",
        "\n".join(errors)
    )


async def run_full_validation(
    user: Any,
    spec_dict: Dict[str, Any],
) -> ValidationResult:
    """
    Run the complete workflow validation pipeline.

    This is the single golden validation path. All callers (MCP tools, Nexus agent)
    should use this function to ensure consistent validation behavior.

    Pipeline steps:
        1. Pydantic schema parse (structural validation gate)
        2. Tool and trigger reference validation
        3. Trigger event_schema auto-fix from registry
        4. Full compilation validation (against the fixed spec)
        5. Re-parse the fixed spec into a final WorkflowSpec model

    Args:
        user: User object for compilation context.
        spec_dict: Raw workflow spec as a dict.

    Returns:
        ValidationResult with success/failure, validated spec, and any auto-fixes.
    """
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports at module level
    from seer.core.compiler.parse import parse_workflow_spec
    from seer.core.errors import ValidationPhaseError
    from seer.tools.trigger_schema_fix import fix_trigger_event_schemas

    # Step 1: Pydantic schema validation (structural gate)
    try:
        parse_workflow_spec(spec_dict)
    except ValidationPhaseError as exc:
        error_msg = str(exc)
        hint = detect_extra_fields(spec_dict, error_msg) or "Check that your spec follows the workflow schema"
        return ValidationResult(
            success=False,
            error=ValidationError("schema_validation", error_msg, hint),
        )

    # Step 1.5: Warn about HITL anti-patterns
    import re as _re  # pylint: disable=import-outside-toplevel  # Reason: Only needed for pattern check
    for node in spec_dict.get("nodes", []):
        if node.get("type") != "hitl":
            continue
        display = node.get("display") or []
        desc = node.get("description") or ""
        if not display and _re.search(r"\$\{", desc):
            node_id = node.get("id", "?")
            return ValidationResult(
                success=False,
                error=ValidationError(
                    "hitl_display",
                    f"HITL node '{node_id}' has ${{...}} expressions in description but empty display array",
                    "Move dynamic content to the display array as {label, value} pairs. "
                    "The description field is a static label and does not evaluate expressions.",
                ),
            )

    # Step 2: Tool and trigger reference validation
    ref_errors = validate_tools_and_triggers(spec_dict)
    if ref_errors:
        return ValidationResult(
            success=False,
            error=ValidationError(
                "reference_validation",
                "Workflow references non-existent tools or triggers",
                "\n".join(ref_errors) + "\nUse search_tools() and list_triggers() to find valid names",
            ),
        )

    # Step 2.5: Trigger provider_config validation
    config_errors = validate_trigger_provider_configs(spec_dict)
    if config_errors:
        return ValidationResult(
            success=False,
            error=ValidationError(
                "provider_config_validation",
                "Trigger configuration is invalid",
                "\n".join(config_errors),
            ),
        )

    # Step 3: Auto-fix trigger event_schemas with canonical schemas from registry
    fixed_spec_dict, schema_fixes = fix_trigger_event_schemas(spec_dict)

    # Step 4: Full compilation validation (always detailed, against the fixed spec)
    compilation_error = await validate_compilation(user, fixed_spec_dict, detailed_errors=True)
    if compilation_error:
        return ValidationResult(
            success=False,
            error=compilation_error,
            schema_fixes=schema_fixes,
        )

    # Step 5: Re-parse fixed spec to get the final WorkflowSpec model
    validated_spec = parse_workflow_spec(fixed_spec_dict)

    return ValidationResult(
        success=True,
        validated_spec=validated_spec,
        fixed_spec_dict=fixed_spec_dict,
        schema_fixes=schema_fixes,
    )
