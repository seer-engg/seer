"""
GitHub tools - commits, pull requests, and repository operations.

All GitHub tools are registered here for easy import and initialization.
"""

from seer.tools.base import register_tool
from seer.tools.github.commits import (
    GitHubListCommitsTool,
    GitHubGetCommitTool,
)
from seer.tools.github.pull_requests import (
    GitHubListPullRequestsTool,
    GitHubGetPullRequestTool,
    GitHubGetPRFilesTool,
)


def register_github_tools():
    """Register all GitHub tools with the tool registry."""
    register_tool(GitHubListCommitsTool())
    register_tool(GitHubGetCommitTool())
    register_tool(GitHubListPullRequestsTool())
    register_tool(GitHubGetPullRequestTool())
    register_tool(GitHubGetPRFilesTool())


__all__ = [
    "register_github_tools",
    "GitHubListCommitsTool",
    "GitHubGetCommitTool",
    "GitHubListPullRequestsTool",
    "GitHubGetPullRequestTool",
    "GitHubGetPRFilesTool",
]
