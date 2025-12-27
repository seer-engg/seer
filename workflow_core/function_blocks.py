"""
Reusable schema definitions for built-in workflow function blocks.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .schema import BlockType


class FunctionBlockSchema(BaseModel):
    """Metadata describing a built-in function-style workflow block."""

    type: BlockType = Field(..., description="Block type identifier")
    label: str = Field(..., description="Human-friendly display name")
    category: str = Field(default="functions", description="Block category grouping")
    description: str = Field(..., description="Summary of what the block does")
    defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default config values applied when block is created",
    )
    config_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema describing config fields and validation rules",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Optional keywords to aid search/filtering",
    )


def _llm_block_schema() -> FunctionBlockSchema:
    return FunctionBlockSchema(
        type=BlockType.LLM,
        label="LLM",
        description="Call a large language model with optional structured output.",
        defaults={
            "system_prompt": "",
            "user_prompt": "Enter your prompt here",
            "model": "gpt-5-mini",
            "temperature": 0,
        },
        config_schema={
            "type": "object",
            "required": ["user_prompt"],
            "properties": {
                "system_prompt": {
                    "type": "string",
                    "title": "System Prompt",
                    "description": "Sets behavior and context for the assistant.",
                    "default": "",
                },
                "user_prompt": {
                    "type": "string",
                    "title": "User Prompt",
                    "description": "Primary instruction sent to the model.",
                    "minLength": 1,
                },
                "model": {
                    "type": "string",
                    "title": "Model",
                    "description": "Model identifier to invoke.",
                    "enum": ["gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-4o"],
                    "default": "gpt-5-mini",
                },
                "temperature": {
                    "type": "number",
                    "title": "Temperature",
                    "description": "Randomness of the response (0-2).",
                    "minimum": 0,
                    "maximum": 2,
                    "default": 0.2,
                },
                "output_schema": {
                    "type": ["object", "null"],
                    "title": "Structured Output Schema",
                    "description": "Optional JSON schema for enforcing structured responses.",
                },
            },
            "additionalProperties": True,
        },
        tags=["generation", "language"],
    )


def _if_else_block_schema() -> FunctionBlockSchema:
    return FunctionBlockSchema(
        type=BlockType.IF_ELSE,
        label="If / Else",
        description="Branch execution based on a boolean expression.",
        defaults={
            "condition": "",
        },
        config_schema={
            "type": "object",
            "required": ["condition"],
            "properties": {
                "condition": {
                    "type": "string",
                    "title": "Condition Expression",
                    "description": "Any Python-style boolean expression (e.g., len(emails) > 0).",
                    "minLength": 1,
                },
            },
            "additionalProperties": True,
        },
        tags=["logic", "branching"],
    )


def _for_loop_block_schema() -> FunctionBlockSchema:
    return FunctionBlockSchema(
        type=BlockType.FOR_LOOP,
        label="For Loop",
        description="Iterate over each item from an array (literal or variable reference) and execute the loop branch before exiting.",
        defaults={
            "array_mode": "variable",
            "array_variable": "items",
            "array_literal": [],
            "item_var": "item",
        },
        config_schema={
            "type": "object",
            "required": ["item_var"],
            "properties": {
                "array_mode": {
                    "type": "string",
                    "title": "Array Source",
                    "enum": ["variable", "literal"],
                    "default": "variable",
                    "description": "Choose whether to resolve items from a variable reference or enter them manually.",
                },
                "array_variable": {
                    "type": "string",
                    "title": "Array Variable",
                    "description": "Variable (e.g., previous block alias) that resolves to the list to iterate.",
                    "minLength": 1,
                    "default": "items",
                },
                "array_literal": {
                    "type": "array",
                    "title": "Manual Items",
                    "description": "Explicit list of items to iterate when not using a variable.",
                    "items": {},
                    "default": [],
                },
                "item_var": {
                    "type": "string",
                    "title": "Item Variable",
                    "description": "Name used for each element inside the loop.",
                    "minLength": 1,
                    "default": "item",
                },
            },
            "additionalProperties": True,
        },
        tags=["logic", "loop"],
    )


def _variable_block_schema() -> FunctionBlockSchema:
    return FunctionBlockSchema(
        type=BlockType.VARIABLE,
        label="Variable",
        description="Capture a literal string, number, or array value for reuse elsewhere in the workflow.",
        defaults={
            "input_type": "string",
            "input": "",
        },
        config_schema={
            "type": "object",
            "required": ["input"],
            "properties": {
                "input_type": {
                    "type": "string",
                    "enum": ["string", "number", "array"],
                    "default": "string",
                    "title": "Input Type",
                    "description": "Controls whether the stored value is treated as a string, number, or array.",
                },
                "input": {
                    "type": ["string", "number", "array"],
                    "title": "Value",
                    "description": "Literal value that will be stored and referenced by downstream blocks.",
                },
            },
            "additionalProperties": True,
        },
        tags=["state", "constants"],
    )


FUNCTION_BLOCK_SCHEMAS: List[FunctionBlockSchema] = [
    _llm_block_schema(),
    _if_else_block_schema(),
    _for_loop_block_schema(),
    _variable_block_schema(),
]


def get_function_block_schemas() -> List[FunctionBlockSchema]:
    """Return all available function block schemas."""
    return FUNCTION_BLOCK_SCHEMAS


__all__ = ["FunctionBlockSchema", "FUNCTION_BLOCK_SCHEMAS", "get_function_block_schemas"]

