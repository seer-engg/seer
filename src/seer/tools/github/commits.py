"""
GitHub commit operations - listing commits and getting commit details with diffs.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.github.base import GitHubAPIClient

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.github.commits")


class GitHubListCommitsTool(GitHubAPIClient):
    """List commits in a GitHub repository with optional filters."""

    name = "github_list_commits"
    description = """List commits in a GitHub repository. Supports filtering by branch, author, and date range.
    Use 'since' and 'until' parameters with ISO 8601 format (e.g., '2024-01-14T00:00:00Z') to filter by date."""
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
                "sha": {
                    "type": "string",
                    "description": "Branch name or commit SHA to start listing from. Defaults to the default branch.",
                },
                "author": {
                    "type": "string",
                    "description": "GitHub username or email to filter commits by author",
                },
                "since": {
                    "type": "string",
                    "description": "Only commits after this date (ISO 8601 format, e.g., '2024-01-14T00:00:00Z')",
                },
                "until": {
                    "type": "string",
                    "description": "Only commits before this date (ISO 8601 format, e.g., '2024-01-21T23:59:59Z')",
                },
                "per_page": {
                    "type": "integer",
                    "description": "Number of commits per page (max 100)",
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
                    "sha": {"type": "string", "description": "Commit SHA"},
                    "commit": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Commit message"},
                            "author": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "email": {"type": "string"},
                                    "date": {"type": "string"},
                                },
                            },
                            "committer": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "email": {"type": "string"},
                                    "date": {"type": "string"},
                                },
                            },
                        },
                    },
                    "author": {
                        "type": "object",
                        "description": "GitHub user who authored the commit",
                        "properties": {
                            "login": {"type": "string"},
                            "avatar_url": {"type": "string"},
                        },
                    },
                    "html_url": {"type": "string", "description": "URL to view commit on GitHub"},
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
        params: Dict[str, Any] = {}

        if sha := arguments.get("sha"):
            params["sha"] = sha
        if author := arguments.get("author"):
            params["author"] = author
        if since := arguments.get("since"):
            params["since"] = since
        if until := arguments.get("until"):
            params["until"] = until

        params["per_page"] = min(arguments.get("per_page", 30), 100)
        params["page"] = arguments.get("page", 1)

        logger.info(
            "Listing GitHub commits: owner=%s, repo=%s, author=%s, since=%s, until=%s",
            owner, repo, arguments.get("author"), arguments.get("since"), arguments.get("until")
        )

        return await self._make_request(
            "GET",
            f"repos/{owner}/{repo}/commits",
            credentials=credentials,
            params=params,
        )


class GitHubGetCommitTool(GitHubAPIClient):
    """Get a specific commit with full details including file changes and diffs."""

    name = "github_get_commit"
    description = """Get detailed information about a specific commit, including the list of files changed and their diffs/patches.
    Use this to analyze what changed in a particular commit."""
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
                "sha": {
                    "type": "string",
                    "description": "Commit SHA (full or abbreviated)",
                },
            },
            "required": ["owner", "repo", "sha"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sha": {"type": "string", "description": "Full commit SHA"},
                "commit": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message"},
                        "author": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                                "date": {"type": "string"},
                            },
                        },
                    },
                },
                "stats": {
                    "type": "object",
                    "description": "Commit statistics",
                    "properties": {
                        "additions": {"type": "integer"},
                        "deletions": {"type": "integer"},
                        "total": {"type": "integer"},
                    },
                },
                "files": {
                    "type": "array",
                    "description": "List of files changed in this commit",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "File path"},
                            "status": {"type": "string", "description": "Change type: added, removed, modified, renamed"},
                            "additions": {"type": "integer"},
                            "deletions": {"type": "integer"},
                            "changes": {"type": "integer"},
                            "patch": {"type": "string", "description": "Unified diff patch for this file"},
                        },
                    },
                },
                "html_url": {"type": "string", "description": "URL to view commit on GitHub"},
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
        sha = str(arguments.get("sha") or "").strip()

        if not owner:
            raise HTTPException(status_code=400, detail="Parameter 'owner' is required")
        if not repo:
            raise HTTPException(status_code=400, detail="Parameter 'repo' is required")
        if not sha:
            raise HTTPException(status_code=400, detail="Parameter 'sha' is required")

        logger.info("Getting GitHub commit: owner=%s, repo=%s, sha=%s", owner, repo, sha)

        return await self._make_request(
            "GET",
            f"repos/{owner}/{repo}/commits/{sha}",
            credentials=credentials,
        )
