"""
Shared workflow validation utilities.

Used by both Nexus agent tools and MCP tools to validate workflow specs.
Provides consistent validation logic for tool references, trigger references,
and full compilation validation.
"""

from __future__ import annotations

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
        hint = "Check that all ${...} references point to valid variables." if detailed_errors else None
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
