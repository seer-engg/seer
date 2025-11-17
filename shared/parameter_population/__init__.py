"""
Dynamic parameter population for MCP tools.

Public API for aggressively populating tool parameters from context.
Works with any tool/service using schema-driven inference.
"""
from shared.parameter_population.context_extraction import (
    extract_all_context_variables,
    format_context_variables_for_llm,
)

__all__ = [
    "extract_all_context_variables",
    "format_context_variables_for_llm",
]

