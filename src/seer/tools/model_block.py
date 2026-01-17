"""
Model block tool for running LLM inference.

Uses LangChain/LangGraph to run model inference with optional structured output.
"""
import json
from typing import Any, Dict, Optional

from fastapi import HTTPException
from langchain_core.messages import HumanMessage

from seer.llm import get_llm
from seer.logger import get_logger
from seer.tools.base import BaseTool, register_tool

logger = get_logger("shared.tools.model_block")


class ModelBlockTool(BaseTool):
    """
    Tool for running LLM inference with optional structured output.

    No OAuth required - uses OpenAI API key from config.
    """

    name = "model_block"
    description = "Run LLM inference with a prompt. Supports structured output via JSON schema."
    required_scopes = []  # No OAuth needed

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for model block tool parameters."""
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Input prompt for the model"
                },
                "model_name": {
                    "type": "string",
                    "description": "Model name (default: from config)",
                    "default": None
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature for generation (0.0-2.0)",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 0.2
                },
                "output_schema": {
                    "type": "object",
                    "description": "Optional JSON schema for structured output",
                    "default": None
                },
                "system_message": {
                    "type": "string",
                    "description": "Optional system message",
                    "default": None
                }
            },
            "required": ["prompt"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model block tool.

        Args:
            access_token: Not used (no OAuth required)
            arguments: Tool arguments

        Returns:
            Dict with "output" (text) and optionally "structured_output" (dict)
        """
        prompt = arguments.get("prompt")
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="'prompt' parameter is required"
            )

        model_name = arguments.get("model_name")
        temperature = arguments.get("temperature", 0.2)
        output_schema = arguments.get("output_schema")
        system_message = arguments.get("system_message")

        try:
            # Get LLM instance
            llm = get_llm(
                model=model_name or None,  # Use default if not specified
                temperature=temperature
            )

            # Build messages
            messages = []
            if system_message:
                from langchain_core.messages import SystemMessage
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))

            # If output_schema provided, use structured output
            if output_schema:
                logger.info("Running model with structured output schema")

                # Use structured output with JSON schema
                # LangChain's with_structured_output supports dict schemas with method="json_schema"
                try:
                    structured_llm = llm.with_structured_output(
                        output_schema,
                        method="json_schema"
                    )
                    result = structured_llm.invoke(messages)

                    # Convert result to dict if it's a Pydantic model
                    if hasattr(result, "model_dump"):
                        structured_output = result.model_dump()
                    elif isinstance(result, dict):
                        structured_output = result
                    else:
                        structured_output = {"result": str(result)}

                    return {
                        "output": json.dumps(structured_output, indent=2),
                        "structured_output": structured_output
                    }
                except Exception as e:
                    logger.warning("Structured output failed, falling back to text: %s", e)
                    # Fall back to regular text generation
                    result = llm.invoke(messages)
                    output_text = result.content if hasattr(result, "content") else str(result)
                    return {
                        "output": output_text,
                        "structured_output": None
                    }
            else:
                # Regular text generation
                logger.info("Running model inference: model=%s, temperature={temperature}", model_name or 'default')
                result = llm.invoke(messages)
                output_text = result.content if hasattr(result, "content") else str(result)

                return {
                    "output": output_text,
                    "structured_output": None
                }

        except ValueError as e:
            # Likely missing API key
            logger.error("Model execution error: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Model execution failed: {str(e)}"
            )
        except Exception as e:
            logger.exception("Unexpected error in model block: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Model execution error: {str(e)}"
            )


# Register the tool
_model_tool = ModelBlockTool()
register_tool(_model_tool)
