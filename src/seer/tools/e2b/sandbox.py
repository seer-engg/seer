"""
E2B persistent sandbox tools.

For agent nodes that need stateful sandbox sessions:
- Create sandbox, get ID
- Run code in existing sandbox
- Kill sandbox when done
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.config import config
from seer.logger import get_logger
from seer.tools.e2b.base import E2BToolBase, serialize_e2b_result

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.e2b.sandbox")


class E2BCreateSandboxTool(E2BToolBase):
    """Create a persistent E2B sandbox for multi-step agent use."""

    name = "codebox_create_sandbox"
    description = (
        "Create a persistent cloud sandbox for code execution. "
        "Returns a sandbox_id that can be used for multiple e2b_run_in_sandbox calls. "
        "The sandbox persists until explicitly killed or times out. "
        "Use for stateful workflows or agent nodes that need multiple executions."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "integer",
                    "description": "Sandbox timeout in seconds (default: 300, max: 3600).",
                    "default": 300,
                    "minimum": 60,
                    "maximum": 3600,
                },
            },
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sandbox_id": {"type": "string", "description": "Sandbox identifier for subsequent calls"},
                "timeout_seconds": {"type": "integer", "description": "Sandbox timeout in seconds"},
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
        _ = access_token, credentials, context

        timeout = min(arguments.get("timeout", config.e2b_sandbox_timeout_seconds), 3600)

        sandbox = await self._create_sandbox(timeout=timeout)

        # Don't kill - leave running for subsequent calls
        return {
            "sandbox_id": sandbox.sandbox_id,
            "timeout_seconds": timeout,
        }


class E2BRunInSandboxTool(E2BToolBase):
    """Run code in an existing persistent sandbox."""

    name = "codebox_run_in_sandbox"
    description = (
        "Execute code in an existing sandbox created by codebox_create_sandbox. "
        "Use the sandbox_id from a previous create_sandbox call. "
        "State persists between calls (variables, files, installed packages)."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sandbox_id": {
                    "type": "string",
                    "description": "Sandbox ID from e2b_create_sandbox",
                },
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
            },
            "required": ["sandbox_id", "code"],
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
                    "description": "Execution results",
                    "items": {"type": "object"},
                },
                "error": {"type": ["string", "null"], "description": "Error message if execution failed"},
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
        _ = access_token, credentials, context

        sandbox_id: str = arguments["sandbox_id"]
        code: str = arguments["code"]

        sandbox = await self._connect_sandbox(sandbox_id)

        try:
            execution = await sandbox.run_code(code)

            results = [serialize_e2b_result(r) for r in execution.results]

            # Extract logs - E2B returns stdout/stderr as lists of strings
            stdout = ""
            stderr = ""
            if hasattr(execution, "logs"):
                logs = execution.logs
                if hasattr(logs, "stdout") and logs.stdout:
                    stdout = "".join(logs.stdout) if isinstance(logs.stdout, list) else logs.stdout
                if hasattr(logs, "stderr") and logs.stderr:
                    stderr = "".join(logs.stderr) if isinstance(logs.stderr, list) else logs.stderr

            return {
                "success": execution.error is None,
                "stdout": stdout,
                "stderr": stderr,
                "results": results,
                "error": str(execution.error) if execution.error else None,
            }
        except Exception as e:  # pylint: disable=broad-exception-caught  # E2B SDK raises undocumented exception types; return structured error
            logger.exception("Run in sandbox failed")
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "results": [],
                "error": str(e),
            }


class E2BKillSandboxTool(E2BToolBase):
    """Terminate a persistent sandbox."""

    name = "codebox_kill_sandbox"
    description = (
        "Terminate a running sandbox. Use when done with a persistent sandbox "
        "to free resources. The sandbox will also auto-terminate after timeout."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sandbox_id": {
                    "type": "string",
                    "description": "Sandbox ID to terminate",
                },
            },
            "required": ["sandbox_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether termination succeeded"},
                "message": {"type": "string", "description": "Status message"},
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
        _ = access_token, credentials, context

        sandbox_id: str = arguments["sandbox_id"]

        try:
            sandbox = await self._connect_sandbox(sandbox_id)
            await sandbox.kill()
            logger.info("Terminated E2B sandbox: %s", sandbox_id)
            return {"success": True, "message": f"Sandbox {sandbox_id} terminated"}
        except Exception as e:  # pylint: disable=broad-exception-caught  # E2B SDK raises undocumented exception types; return structured error
            return {"success": False, "message": str(e)}


__all__ = [
    "E2BCreateSandboxTool",
    "E2BRunInSandboxTool",
    "E2BKillSandboxTool",
]
