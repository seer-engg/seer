"""
Python code execution service for workflow code blocks.
Executes Python code safely with access to workflow context.
"""
import asyncio
import traceback
from typing import Any, Dict, Optional
from shared.logger import get_logger

logger = get_logger("api.workflows.code_executor")


class CodeExecutionError(Exception):
    """Exception raised during code execution."""
    pass


class CodeExecutor:
    """
    Executes Python code blocks safely.
    
    TODO: Integrate with proper sandbox infrastructure when available.
    For now, uses restricted execution environment.
    """
    
    def __init__(self, timeout_seconds: int = 30, max_memory_mb: int = 256):
        """
        Initialize code executor.
        
        Args:
            timeout_seconds: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB (not enforced yet)
        """
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
    
    async def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Python code with workflow context.
        
        Args:
            code: Python code to execute
            context: Workflow context (previous block outputs, variables)
            
        Returns:
            Dictionary with execution results:
            - 'output': The return value or last expression result
            - 'variables': Updated variables from context
            - 'error': Error message if execution failed
            
        Raises:
            CodeExecutionError: If execution fails
        """
        if not code or not code.strip():
            return {'output': None, 'variables': context or {}, 'error': None}
        
        # Prepare execution context
        exec_context = {
            '__builtins__': {
                'abs': abs,
                'all': all,
                'any': any,
                'bool': bool,
                'dict': dict,
                'enumerate': enumerate,
                'float': float,
                'int': int,
                'len': len,
                'list': list,
                'max': max,
                'min': min,
                'range': range,
                'reversed': reversed,
                'round': round,
                'set': set,
                'sorted': sorted,
                'str': str,
                'sum': sum,
                'tuple': tuple,
                'type': type,
                'zip': zip,
                'print': print,  # Allow print for debugging
            },
            '__name__': '__workflow__',
        }
        
        # Add workflow context variables
        if context:
            exec_context.update(context)
        
        # Add common workflow utilities
        exec_context['_result'] = None
        
        try:
            # Execute code with timeout
            result = await asyncio.wait_for(
                self._execute_code(code, exec_context),
                timeout=self.timeout_seconds
            )
            
            # Extract output and variables
            output = exec_context.get('_result')
            variables = {k: v for k, v in exec_context.items() 
                        if not k.startswith('__') and k != '_result'}
            
            return {
                'output': output,
                'variables': variables,
                'error': None,
            }
            
        except asyncio.TimeoutError:
            error_msg = f"Code execution timed out after {self.timeout_seconds} seconds"
            logger.error(error_msg)
            raise CodeExecutionError(error_msg)
            
        except Exception as e:
            error_msg = f"Code execution error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {
                'output': None,
                'variables': context or {},
                'error': error_msg,
            }
    
    async def _execute_code(self, code: str, context: Dict[str, Any]) -> None:
        """
        Execute code in the given context.
        
        Args:
            code: Python code to execute
            context: Execution context dictionary
        """
        # Wrap code to capture result
        wrapped_code = f"""
_result = None
try:
    # User code
    {code}
    # If no explicit return, use last expression or None
    if '_result' not in locals() or _result is None:
        # Try to get last expression result
        pass
except Exception as e:
    _result = {{'error': str(e), 'type': type(e).__name__}}
    raise
"""
        
        # Execute in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, exec, wrapped_code, context)
    
    def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate code syntax without executing.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, '<workflow>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"


# Global executor instance
_executor = CodeExecutor()


async def execute_code_block(
    code: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to execute code block.
    
    Args:
        code: Python code to execute
        context: Workflow context
        
    Returns:
        Execution results dictionary
    """
    return await _executor.execute(code, context)


def validate_code_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    Convenience function to validate code syntax.
    
    Args:
        code: Python code to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    return _executor.validate_code(code)


__all__ = [
    "CodeExecutionError",
    "CodeExecutor",
    "execute_code_block",
    "validate_code_syntax",
]

