"""
E2B sandbox tools package.

Provides secure cloud sandbox code execution using E2B.
Uses platform-owned API key (E2B_API_KEY) - always available for all customers.
"""
from seer.tools.base import register_tool
from seer.tools.e2b.data_loader import E2BLoadDataTool
from seer.tools.e2b.run_code import E2BRunCodeTool
from seer.tools.e2b.sandbox import (
    E2BCreateSandboxTool,
    E2BKillSandboxTool,
    E2BRunInSandboxTool,
)


def register_e2b_tools() -> None:
    """Register all E2B sandbox tools."""
    register_tool(E2BRunCodeTool())
    register_tool(E2BCreateSandboxTool())
    register_tool(E2BRunInSandboxTool())
    register_tool(E2BKillSandboxTool())
    register_tool(E2BLoadDataTool())


__all__ = [
    "E2BRunCodeTool",
    "E2BCreateSandboxTool",
    "E2BRunInSandboxTool",
    "E2BKillSandboxTool",
    "E2BLoadDataTool",
    "register_e2b_tools",
]
