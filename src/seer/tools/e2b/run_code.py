"""
E2B run code tool - ephemeral code execution.

Creates a sandbox, runs code, returns results, destroys sandbox.
Best for simple one-shot code execution in workflow tool nodes.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.logger import get_logger
from seer.tools.e2b.base import E2BToolBase, serialize_e2b_result

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.e2b.run_code")


class E2BRunCodeTool(E2BToolBase):
    """Execute Python code in an ephemeral E2B sandbox."""

    name = "codebox_run_code"
    description = (
        "Execute Python code in a secure cloud sandbox. "
        "The sandbox is created fresh for each execution and destroyed after. "
        "Use for data processing, calculations, file manipulation, or any Python code. "
        "Returns stdout, stderr, and execution results."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute in the sandbox.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (default: 60, max: 300).",
                    "default": 60,
                    "minimum": 1,
                    "maximum": 300,
                },
            },
            "required": ["code"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether execution succeeded"},
                "stdout": {"type": "string", "description": "Standard output from code"},
                "stderr": {"type": "string", "description": "Standard error from code"},
                "results": {
                    "type": "array",
                    "description": "Execution results (for expressions that return values)",
                    "items": {"type": "object"},
                },
                "error": {"type": ["string", "null"], "description": "Error message if execution failed"},
                "execution_time_ms": {"type": "integer", "description": "Execution time in milliseconds"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        # access_token, credentials unused - platform API key from config
        _ = access_token, credentials, context

        code: str = arguments["code"]
        timeout: int = min(arguments.get("timeout", 60), 300)

        start_time = time.perf_counter()

        # Create sandbox with buffer time for setup
        sandbox = await self._create_sandbox(timeout=timeout + 30)

        try:
            execution = await sandbox.run_code(code)

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Format results using E2B SDK attribute inspection
            results = []
            for result in execution.results:
                serialized = serialize_e2b_result(result)
                serialized["is_main_result"] = getattr(result, "is_main_result", False)
                results.append(serialized)

            # Extract logs - E2B returns stdout/stderr as lists of strings
            stdout = ""
            stderr = ""
            if hasattr(execution, "logs"):
                logs = execution.logs
                if hasattr(logs, "stdout") and logs.stdout:
                    stdout = "".join(logs.stdout) if isinstance(logs.stdout, list) else logs.stdout
                if hasattr(logs, "stderr") and logs.stderr:
                    stderr = "".join(logs.stderr) if isinstance(logs.stderr, list) else logs.stderr

            logger.debug(
                "E2B code execution completed: success=%s, time=%dms",
                execution.error is None,
                execution_time_ms,
            )

            return {
                "success": execution.error is None,
                "stdout": stdout,
                "stderr": stderr,
                "results": results,
                "error": str(execution.error) if execution.error else None,
                "execution_time_ms": execution_time_ms,
            }

        except Exception as e:  # pylint: disable=broad-exception-caught  # E2B SDK raises undocumented exception types; return structured error
            logger.exception("Code execution failed")
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "results": [],
                "error": f"Execution error: {e!s}",
                "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
            }
        finally:
            try:
                await sandbox.kill()
            except Exception:  # pylint: disable=broad-exception-caught  # E2B SDK raises undocumented exception types; best-effort cleanup
                logger.warning("Failed to kill sandbox (may have timed out)")


__all__ = ["E2BRunCodeTool"]
