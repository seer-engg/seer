"""
Dynamic parameter population for MCP tools.

Public API for aggressively populating tool parameters from context.
Works with any tool/service using schema-driven inference.
"""
from shared.parameter_population.context_extraction import (
    extract_all_context_variables,
    format_context_variables_for_llm,
)
from shared.parameter_population.completion import (
    complete_action_parameters,
    complete_action_list,
    ParameterCompletionError,
)

__all__ = [
    "extract_all_context_variables",
    "format_context_variables_for_llm",
    "complete_action_parameters",
    "complete_action_list",
    "ParameterCompletionError",
]

