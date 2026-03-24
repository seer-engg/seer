"""
GitHub pull request operations - listing PRs, getting PR details, and fetching PR files/diffs.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.github.base import GitHubAPIClient

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.github.pull_requests")


class GitHubListPullRequestsTool(GitHubAPIClient):
    """List pull requests for a GitHub repository."""

    name = "github_list_pull_requests"
    description = """List pull requests for a repository. Filter by state (open, closed, all).
    Returns basic PR info including title, number, author, and dates."""
    required_scopes = ["repo"]
    integration_type = "github"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (username or organization)",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name",
                },
                "state": {
                    "type": "string",
                    "description": "Filter by PR state",
                    "enum": ["open", "closed", "all"],
                    "default": "open",
                },
                "head": {
                    "type": "string",
                    "description": "Filter by head branch (format: 'user:branch' or 'branch')",
                },
                "base": {
                    "type": "string",
                    "description": "Filter by base branch name",
                },
                "sort": {
                    "type": "string",
                    "description": "Sort by field",
                    "enum": ["created", "updated", "popularity", "long-running"],
                    "default": "created",
                },
                "direction": {
                    "type": "string",
                    "description": "Sort direction",
                    "enum": ["asc", "desc"],
                    "default": "desc",
                },
                "per_page": {
                    "type": "integer",
                    "description": "Number of PRs per page (max 100)",
                    "default": 30,
                    "maximum": 100,
                },
                "page": {
                    "type": "integer",
                    "description": "Page number for pagination",
                    "default": 1,
                },
            },
            "required": ["owner", "repo"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer", "description": "PR number"},
                    "title": {"type": "string", "description": "PR title"},
                    "state": {"type": "string", "description": "PR state (open/closed)"},
                    "user": {
                        "type": "object",
                        "properties": {
                            "login": {"type": "string"},
                            "avatar_url": {"type": "string"},
                        },
                    },
                    "body": {"type": "string", "description": "PR description"},
                    "created_at": {"type": "string"},
                    "updated_at": {"type": "string"},
                    "merged_at": {"type": "string"},
                    "html_url": {"type": "string", "description": "URL to view PR on GitHub"},
                    "head": {
                        "type": "object",
                        "properties": {
                            "ref": {"type": "string", "description": "Head branch name"},
                            "sha": {"type": "string", "description": "Head commit SHA"},
                        },
                    },
                    "base": {
                        "type": "object",
                        "properties": {
                            "ref": {"type": "string", "description": "Base branch name"},
                        },
                    },
                },
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        _ = access_token, context  # unused but required for interface consistency

        owner = str(arguments.get("owner") or "").strip()
        repo = str(arguments.get("repo") or "").strip()

        if not owner:
            raise HTTPException(status_code=400, detail="Parameter 'owner' is required")
        if not repo:
            raise HTTPException(status_code=400, detail="Parameter 'repo' is required")

        # Build query parameters
        params: Dict[str, Any] = {
            "state": arguments.get("state", "open"),
            "sort": arguments.get("sort", "created"),
            "direction": arguments.get("direction", "desc"),
            "per_page": min(arguments.get("per_page", 30), 100),
            "page": arguments.get("page", 1),
        }

        if head := arguments.get("head"):
            params["head"] = head
        if base := arguments.get("base"):
            params["base"] = base

        logger.info("Listing GitHub PRs: owner=%s, repo=%s, state=%s", owner, repo, params["state"])

        return await self._make_request(
            "GET",
            f"repos/{owner}/{repo}/pulls",
            credentials=credentials,
            params=params,
        )


class GitHubGetPullRequestTool(GitHubAPIClient):
    """Get detailed information about a specific pull request."""

    name = "github_pull_request_read_get"
    description = """Get detailed information about a specific pull request including title, description,
    author, reviewers, labels, and merge status."""
    required_scopes = ["repo"]
    integration_type = "github"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (username or organization)",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name",
                },
                "pullNumber": {
                    "type": "integer",
                    "description": "Pull request number",
                },
                "pull_number": {
                    "type": "integer",
                    "description": "Pull request number (alias for pullNumber)",
                },
            },
            "required": ["owner", "repo"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "number": {"type": "integer", "description": "PR number"},
                "title": {"type": "string", "description": "PR title"},
                "state": {"type": "string", "description": "PR state (open/closed)"},
                "body": {"type": "string", "description": "PR description"},
                "user": {
                    "type": "object",
                    "properties": {
                        "login": {"type": "string"},
                    },
                },
                "merged": {"type": "boolean", "description": "Whether the PR was merged"},
                "mergeable": {"type": "boolean", "description": "Whether the PR can be merged"},
                "additions": {"type": "integer", "description": "Lines added"},
                "deletions": {"type": "integer", "description": "Lines deleted"},
                "changed_files": {"type": "integer", "description": "Number of files changed"},
                "html_url": {"type": "string", "description": "URL to view PR on GitHub"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        _ = access_token, context  # unused but required for interface consistency

        owner = str(arguments.get("owner") or "").strip()
        repo = str(arguments.get("repo") or "").strip()
        # Support both pullNumber (legacy) and pull_number parameter names
        pull_number = arguments.get("pullNumber") or arguments.get("pull_number")

        if not owner:
            raise HTTPException(status_code=400, detail="Parameter 'owner' is required")
        if not repo:
            raise HTTPException(status_code=400, detail="Parameter 'repo' is required")
        if not pull_number:
            raise HTTPException(status_code=400, detail="Parameter 'pullNumber' or 'pull_number' is required")

        logger.info("Getting GitHub PR: owner=%s, repo=%s, pull_number=%s", owner, repo, pull_number)

        return await self._make_request(
            "GET",
            f"repos/{owner}/{repo}/pulls/{pull_number}",
            credentials=credentials,
        )


class GitHubGetPRFilesTool(GitHubAPIClient):
    """Get the list of files changed in a pull request with diffs."""

    name = "github_get_pr_files"
    description = """Get the list of files changed in a pull request, including the diff/patch for each file.
    Use this to analyze what code changes are proposed in a PR."""
    required_scopes = ["repo"]
    integration_type = "github"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (username or organization)",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name",
                },
                "pull_number": {
                    "type": "integer",
                    "description": "Pull request number",
                },
                "per_page": {
                    "type": "integer",
                    "description": "Number of files per page (max 100)",
                    "default": 30,
                    "maximum": 100,
                },
                "page": {
                    "type": "integer",
                    "description": "Page number for pagination",
                    "default": 1,
                },
            },
            "required": ["owner", "repo", "pull_number"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sha": {"type": "string", "description": "Blob SHA of the file"},
                    "filename": {"type": "string", "description": "File path"},
                    "status": {
                        "type": "string",
                        "description": "Change type",
                        "enum": ["added", "removed", "modified", "renamed", "copied", "changed", "unchanged"],
                    },
                    "additions": {"type": "integer", "description": "Lines added"},
                    "deletions": {"type": "integer", "description": "Lines deleted"},
                    "changes": {"type": "integer", "description": "Total lines changed"},
                    "patch": {"type": "string", "description": "Unified diff patch for this file"},
                    "previous_filename": {"type": "string", "description": "Previous filename if renamed"},
                },
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        _ = access_token, context  # unused but required for interface consistency

        owner = str(arguments.get("owner") or "").strip()
        repo = str(arguments.get("repo") or "").strip()
        pull_number = arguments.get("pull_number")

        if not owner:
            raise HTTPException(status_code=400, detail="Parameter 'owner' is required")
        if not repo:
            raise HTTPException(status_code=400, detail="Parameter 'repo' is required")
        if not pull_number:
            raise HTTPException(status_code=400, detail="Parameter 'pull_number' is required")

        params: Dict[str, Any] = {
            "per_page": min(arguments.get("per_page", 30), 100),
            "page": arguments.get("page", 1),
        }

        logger.info("Getting GitHub PR files: owner=%s, repo=%s, pull_number=%s", owner, repo, pull_number)

        return await self._make_request(
            "GET",
            f"repos/{owner}/{repo}/pulls/{pull_number}/files",
            credentials=credentials,
            params=params,
        )
