"""
Sandboxed execution environment for RLM code using RestrictedPython.

Provides safe Python code execution with restricted builtins and
helper functions for examining and processing context data.
"""

import asyncio
import sys
from io import StringIO
from typing import Any, Dict

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr

from shared.logger import get_logger
from shared.tools.reasoning.helpers import RLMHelpers

logger = get_logger("shared.tools.reasoning.sandbox")

# Memory limits
MAX_CONTEXT_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CHUNK_SIZE = 1 * 1024 * 1024     # 1MB


class RLMSandbox:
    """Sandboxed execution environment for RLM code."""

    def __init__(
        self,
        context: Dict[str, Any],
        model: str,
        max_depth: int,
        timeout: int = 60
    ):
        self.context = context
        self.model = model
        self.max_depth = max_depth
        self.timeout = timeout
        self.helpers = RLMHelpers(context, model, max_depth)

    def _get_safe_builtins(self) -> Dict[str, Any]:
        """
        Return safe builtins allowlist.

        Only includes safe operations - no file I/O, network, imports, or eval.
        """
        return {
            # Type constructors
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            # Iteration and functional
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            # Aggregation
            'min': min,
            'max': max,
            'sum': sum,
            'any': any,
            'all': all,
            'sorted': sorted,
            'reversed': reversed,
            # Constants
            'True': True,
            'False': False,
            'None': None,
            # RestrictedPython guards for safe execution
            '_getitem_': lambda obj, key: obj[key],  # For dict/list access: obj[key]
            '_getiter_': iter,  # For iteration: for x in obj
            '_iter_unpack_sequence_': guarded_iter_unpack_sequence,  # For unpacking: a, b = seq
            '_getattr_': safer_getattr,  # For attribute access: obj.attr
            '_inplacevar_': lambda op, x, y: op(x, y),  # For in-place ops: x += y
            # String operations
            'ord': ord,
            'chr': chr,
            'abs': abs,
            'round': round,
            'dir': dir,
        }

    def _get_safe_globals(self) -> Dict[str, Any]:
        """Build safe global namespace with builtins and helpers."""
        safe_globals = {
            '__builtins__': self._get_safe_builtins(),
            # Helper functions
            'examine': self.helpers.examine,
            'search': self.helpers.search,
            'chunk': self.helpers.chunk,
            'sub_llm': self.helpers.sub_llm,
            # Context access
            'context': self.context,
        }
        return safe_globals

    async def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute restricted Python code with timeout.

        Args:
            code: Python code to execute

        Returns:
            Dictionary with 'result', 'execution_log', and 'stats'

        Raises:
            ValueError: If code compilation fails
            TimeoutError: If execution exceeds timeout
            RecursionError: If recursion depth exceeded
        """
        # Validate context size
        context_size = len(str(self.context))
        if context_size > MAX_CONTEXT_SIZE:
            raise ValueError(
                f"Context size ({context_size} bytes) exceeds maximum "
                f"({MAX_CONTEXT_SIZE} bytes)"
            )

        # Compile code with RestrictedPython
        logger.debug("Compiling RLM code (context size: %d bytes)", context_size)
        try:
            byte_code = compile_restricted(code, '<string>', 'exec')
        except SyntaxError as e:
            logger.error("RLM code compilation failed: %s", str(e))
            raise ValueError(f"Code compilation failed: {e}") from e

        # RestrictedPython 8.x returns code object directly (not CompileResult)
        if byte_code is None:
            raise ValueError("Code compilation failed: returned None")

        # Prepare execution environment
        safe_globals = self._get_safe_globals()
        safe_locals: Dict[str, Any] = {}

        # Capture stdout
        old_stdout = sys.stdout
        stdout_capture = StringIO()
        sys.stdout = stdout_capture

        try:
            # Execute with timeout
            logger.debug("Executing RLM code with %ds timeout", self.timeout)
            await asyncio.wait_for(
                self._run_code_async(byte_code, safe_globals, safe_locals),
                timeout=self.timeout
            )

            # Get result (from 'result' variable or all locals)
            result = safe_locals.get('result', safe_locals)

            # Collect stats
            stats = self.helpers.get_stats()
            stats['execution_time_ms'] = 0  # Will be set by caller

            return {
                'result': result,
                'execution_log': self.helpers.execution_log,
                'stats': stats,
                'stdout': stdout_capture.getvalue(),
            }

        except asyncio.TimeoutError as e:
            logger.error("RLM code execution timeout after %ds", self.timeout)
            raise TimeoutError(
                f"Code execution exceeded {self.timeout}s timeout. "
                f"Consider simplifying the task or increasing timeout."
            ) from e

        except RecursionError as e:
            logger.error("RLM recursion depth exceeded: %s", str(e))
            # Return partial results with error
            result = safe_locals.get('result', safe_locals)
            return {
                'result': result,
                'execution_log': self.helpers.execution_log,
                'stats': self.helpers.get_stats(),
                'error': str(e),
                'stdout': stdout_capture.getvalue(),
            }

        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all user code errors and return partial results with error details
            logger.exception("RLM code execution failed: %s", str(e))
            # Return partial results with error
            result = safe_locals.get('result', safe_locals)
            return {
                'result': result,
                'execution_log': self.helpers.execution_log,
                'stats': self.helpers.get_stats(),
                'error': str(e),
                'stdout': stdout_capture.getvalue(),
            }

        finally:
            sys.stdout = old_stdout

    async def _run_code_async(
        self,
        byte_code: Any,
        safe_globals: Dict[str, Any],
        safe_locals: Dict[str, Any]
    ) -> None:
        """
        Run compiled code in thread pool to avoid blocking.

        RestrictedPython exec is synchronous, so we run it in a thread.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            exec,  # pylint: disable=exec-used # Reason: Using RestrictedPython compile_restricted for sandboxing
            byte_code,
            safe_globals,
            safe_locals
        )
