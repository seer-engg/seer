"""
Recursive Language Model (RLM) tool for Seer.

Enables multi-step complex reasoning by providing a sandboxed Python environment
where LLMs can write code to examine, decompose, and recursively process large contexts.

Based on: https://arxiv.org/html/2512.24601v1
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from shared.logger import get_logger
from shared.tools.base import BaseTool
from shared.tools.reasoning.sandbox import RLMSandbox

logger = get_logger("shared.tools.reasoning.rlm_tool")


class RecursiveLanguageModelTool(BaseTool):
    """
    Execute Python code to decompose and reason over large contexts recursively.

    This tool provides a sandboxed Python environment with helper functions:
    - examine(data): Inspect data structure without printing full content
    - search(data, pattern/key/value): Filter data by criteria
    - chunk(data, size, overlap): Split data into manageable chunks
    - sub_llm(prompt, context_chunk, system): Make recursive LLM calls

    The tool enables multi-step complex reasoning by breaking down tasks into
    subtasks and recursively processing chunks of large context.
    """

    name = "recursive_language_model"
    description = (
        "Execute Python code to decompose and reason over large contexts recursively. "
        "Provides helper functions: examine(data), search(data, pattern), "
        "chunk(data, size), sub_llm(prompt, context_chunk). "
        "Use for complex multi-step analysis, large document processing, "
        "or recursive decomposition of complex tasks."
    )
    required_scopes = []  # No OAuth needed
    integration_type = None  # Internal reasoning tool

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute in sandbox. Available functions: "
                        "examine(data) - inspect structure, "
                        "search(data, pattern/key/value) - filter data, "
                        "chunk(data, size, overlap) - split into chunks, "
                        "sub_llm(prompt, context_chunk, system) - recursive LLM call. "
                        "Context available as 'context' variable. "
                        "Store final result in 'result' variable."
                    )
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Context data accessible as 'context' variable in code. "
                        "Can be any JSON-serializable data structure. "
                        "Maximum size: 10MB."
                    )
                },
                "model": {
                    "type": "string",
                    "description": "LLM model for recursive sub_llm calls",
                    "default": "gpt-4o-mini"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum recursion depth for sub_llm calls",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds",
                    "default": 60,
                    "minimum": 10,
                    "maximum": 300
                }
            },
            "required": ["code", "context"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "result": {
                    "description": "Execution result from code (value of 'result' variable or all locals)"
                },
                "execution_log": {
                    "type": "array",
                    "description": "Trace of recursive sub_llm calls",
                    "items": {
                        "type": "object",
                        "properties": {
                            "depth": {"type": "integer"},
                            "operation": {"type": "string"},
                            "prompt_preview": {"type": "string"},
                            "context_size": {"type": "integer"},
                            "duration_ms": {"type": "number"},
                            "timestamp": {"type": "string"}
                        }
                    }
                },
                "stats": {
                    "type": "object",
                    "description": "Execution statistics",
                    "properties": {
                        "total_llm_calls": {"type": "integer"},
                        "max_depth_reached": {"type": "integer"},
                        "execution_time_ms": {"type": "number"}
                    }
                },
                "stdout": {
                    "type": "string",
                    "description": "Captured stdout from code execution"
                },
                "error": {
                    "type": "string",
                    "description": "Error message if execution failed (partial results still returned)"
                }
            },
            "required": ["result", "execution_log", "stats"]
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute RLM code in sandboxed environment.

        Args:
            access_token: Not used (no OAuth required)
            arguments: Tool arguments (code, context, model, max_depth, timeout)

        Returns:
            Dictionary with result, execution_log, stats, stdout, and optional error

        Raises:
            ValueError: If required parameters missing or invalid
        """
        # Validate required parameters
        code = arguments.get("code")
        context = arguments.get("context")

        if not code:
            raise ValueError("Parameter 'code' is required")
        if context is None:
            raise ValueError("Parameter 'context' is required")
        if not isinstance(code, str):
            raise ValueError("Parameter 'code' must be a string")
        if not isinstance(context, dict):
            raise ValueError("Parameter 'context' must be an object/dict")

        # Get optional parameters
        model = arguments.get("model", "gpt-4o-mini")
        max_depth = arguments.get("max_depth", 3)
        timeout = arguments.get("timeout", 60)

        # Validate ranges
        if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 10:
            raise ValueError("Parameter 'max_depth' must be an integer between 1 and 10")
        if not isinstance(timeout, int) or timeout < 10 or timeout > 300:
            raise ValueError("Parameter 'timeout' must be an integer between 10 and 300")

        logger.info(
            "Executing RLM tool: model=%s, max_depth=%d, timeout=%ds, context_size=%d bytes",
            model,
            max_depth,
            timeout,
            len(str(context))
        )

        # Create sandbox and execute
        sandbox = RLMSandbox(
            context=context,
            model=model,
            max_depth=max_depth,
            timeout=timeout
        )

        start_time = datetime.now(timezone.utc)
        try:
            result = await sandbox.execute(code)

            # Add total execution time
            end_time = datetime.now(timezone.utc)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            result['stats']['execution_time_ms'] = round(execution_time_ms, 2)

            logger.info(
                "RLM execution completed: %d LLM calls, max depth %d, %dms total",
                result['stats']['total_llm_calls'],
                result['stats']['max_depth_reached'],
                result['stats']['execution_time_ms']
            )

            return result

        except (ValueError, TimeoutError) as e:
            # Re-raise validation and timeout errors
            logger.error("RLM execution failed: %s", str(e))
            raise

        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all sandbox errors and return structured response with error details
            # Catch-all for unexpected errors
            logger.exception("RLM execution encountered unexpected error: %s", str(e))
            end_time = datetime.now(timezone.utc)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000

            return {
                'result': None,
                'execution_log': [],
                'stats': {
                    'total_llm_calls': 0,
                    'max_depth_reached': 0,
                    'execution_time_ms': round(execution_time_ms, 2)
                },
                'stdout': '',
                'error': str(e)
            }
