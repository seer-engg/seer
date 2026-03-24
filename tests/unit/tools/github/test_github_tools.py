"""Unit tests for GitHub tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from seer.tools.github.base import GitHubAPIClient
from seer.tools.github.commits import GitHubListCommitsTool, GitHubGetCommitTool
from seer.tools.github.pull_requests import (
    GitHubListPullRequestsTool,
    GitHubGetPullRequestTool,
    GitHubGetPRFilesTool,
)
from seer.tools.credential_resolver import ResolvedCredentials


def make_credentials(access_token: str = "test_token") -> ResolvedCredentials:
    """Create mock credentials for testing."""
    creds = MagicMock(spec=ResolvedCredentials)
    creds.access_token = access_token
    return creds


@pytest.mark.unit
class TestGitHubAPIClient:
    """Tests for the GitHubAPIClient base class."""

    def test_get_access_token_success(self):
        """Test token extraction from credentials."""
        # Create a concrete implementation for testing
        class TestTool(GitHubAPIClient):
            name = "test_tool"
            description = "Test tool"

            async def execute(self, access_token, arguments, *, credentials=None, context=None):
                return {}

        tool = TestTool()
        creds = make_credentials("my_token")
        token = tool._get_access_token(creds)
        assert token == "my_token"

    def test_get_access_token_missing(self):
        """Test error when no token provided."""
        class TestTool(GitHubAPIClient):
            name = "test_tool"
            description = "Test tool"

            async def execute(self, access_token, arguments, *, credentials=None, context=None):
                return {}

        tool = TestTool()

        with pytest.raises(HTTPException) as exc_info:
            tool._get_access_token(None)

        assert exc_info.value.status_code == 401
        assert "access token" in exc_info.value.detail.lower()

    def test_build_headers(self):
        """Test HTTP headers are built correctly."""
        class TestTool(GitHubAPIClient):
            name = "test_tool"
            description = "Test tool"

            async def execute(self, access_token, arguments, *, credentials=None, context=None):
                return {}

        tool = TestTool()
        headers = tool._build_headers("my_token")

        assert headers["Authorization"] == "Bearer my_token"
        assert "application/vnd.github" in headers["Accept"]
        assert "X-GitHub-Api-Version" in headers


@pytest.mark.unit
class TestGitHubListCommitsTool:
    """Tests for GitHubListCommitsTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = GitHubListCommitsTool()
        assert tool.name == "github_list_commits"
        assert tool.integration_type == "github"
        assert tool.provider == "github"
        assert "repo" in tool.required_scopes

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = GitHubListCommitsTool()
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "owner" in schema["properties"]
        assert "repo" in schema["properties"]
        assert "sha" in schema["properties"]
        assert "author" in schema["properties"]
        assert "since" in schema["properties"]
        assert "until" in schema["properties"]
        assert "per_page" in schema["properties"]
        assert "owner" in schema["required"]
        assert "repo" in schema["required"]

    def test_get_output_schema(self):
        """Test output schema has expected fields."""
        tool = GitHubListCommitsTool()
        schema = tool.get_output_schema()

        assert schema["type"] == "array"
        assert "items" in schema
        assert "sha" in schema["items"]["properties"]
        assert "commit" in schema["items"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_missing_owner(self):
        """Test execute raises error when owner is missing."""
        tool = GitHubListCommitsTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {"repo": "test-repo"}, credentials=make_credentials())

        assert exc_info.value.status_code == 400
        assert "owner" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_execute_missing_repo(self):
        """Test execute raises error when repo is missing."""
        tool = GitHubListCommitsTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {"owner": "octocat"}, credentials=make_credentials())

        assert exc_info.value.status_code == 400
        assert "repo" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful commit listing."""
        tool = GitHubListCommitsTool()

        mock_response = [
            {
                "sha": "abc123",
                "commit": {
                    "message": "Initial commit",
                    "author": {"name": "Test User", "email": "test@example.com", "date": "2024-01-15T10:00:00Z"},
                },
                "html_url": "https://github.com/octocat/Hello-World/commit/abc123",
            }
        ]

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                None,
                {"owner": "octocat", "repo": "Hello-World"},
                credentials=make_credentials(),
            )

            assert len(result) == 1
            assert result[0]["sha"] == "abc123"
            mock_request.assert_called_once()

            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"
            assert "repos/octocat/Hello-World/commits" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_execute_with_date_filter(self):
        """Test commit listing with since/until parameters."""
        tool = GitHubListCommitsTool()

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = []

            await tool.execute(
                None,
                {
                    "owner": "octocat",
                    "repo": "Hello-World",
                    "since": "2024-01-14T00:00:00Z",
                    "until": "2024-01-21T23:59:59Z",
                },
                credentials=make_credentials(),
            )

            call_args = mock_request.call_args
            params = call_args[1].get("params", {})
            assert params["since"] == "2024-01-14T00:00:00Z"
            assert params["until"] == "2024-01-21T23:59:59Z"

    @pytest.mark.asyncio
    async def test_execute_with_author_filter(self):
        """Test commit listing with author parameter."""
        tool = GitHubListCommitsTool()

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = []

            await tool.execute(
                None,
                {"owner": "octocat", "repo": "Hello-World", "author": "octocat"},
                credentials=make_credentials(),
            )

            call_args = mock_request.call_args
            params = call_args[1].get("params", {})
            assert params["author"] == "octocat"


@pytest.mark.unit
class TestGitHubGetCommitTool:
    """Tests for GitHubGetCommitTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = GitHubGetCommitTool()
        assert tool.name == "github_get_commit"
        assert tool.integration_type == "github"
        assert "repo" in tool.required_scopes

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = GitHubGetCommitTool()
        schema = tool.get_parameters_schema()

        assert "owner" in schema["properties"]
        assert "repo" in schema["properties"]
        assert "sha" in schema["properties"]
        assert "sha" in schema["required"]

    def test_get_output_schema(self):
        """Test output schema has files array."""
        tool = GitHubGetCommitTool()
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "files" in schema["properties"]
        assert "stats" in schema["properties"]

    @pytest.mark.asyncio
    async def test_execute_missing_sha(self):
        """Test execute raises error when sha is missing."""
        tool = GitHubGetCommitTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                None,
                {"owner": "octocat", "repo": "Hello-World"},
                credentials=make_credentials(),
            )

        assert exc_info.value.status_code == 400
        assert "sha" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_execute_with_diff(self):
        """Test getting commit with files array returned."""
        tool = GitHubGetCommitTool()

        mock_response = {
            "sha": "abc123def456",
            "commit": {
                "message": "Add new feature",
                "author": {"name": "Test", "email": "test@example.com", "date": "2024-01-15T10:00:00Z"},
            },
            "stats": {"additions": 10, "deletions": 5, "total": 15},
            "files": [
                {
                    "filename": "src/main.py",
                    "status": "modified",
                    "additions": 10,
                    "deletions": 5,
                    "patch": "@@ -1,5 +1,10 @@\n+new line",
                }
            ],
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                None,
                {"owner": "octocat", "repo": "Hello-World", "sha": "abc123"},
                credentials=make_credentials(),
            )

            assert result["sha"] == "abc123def456"
            assert len(result["files"]) == 1
            assert result["files"][0]["filename"] == "src/main.py"
            assert "patch" in result["files"][0]

            call_args = mock_request.call_args
            assert "repos/octocat/Hello-World/commits/abc123" in call_args[0][1]


@pytest.mark.unit
class TestGitHubListPullRequestsTool:
    """Tests for GitHubListPullRequestsTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = GitHubListPullRequestsTool()
        assert tool.name == "github_list_pull_requests"
        assert tool.integration_type == "github"
        assert tool.provider == "github"

    def test_get_parameters_schema(self):
        """Test parameter schema includes state filter."""
        tool = GitHubListPullRequestsTool()
        schema = tool.get_parameters_schema()

        assert "state" in schema["properties"]
        assert schema["properties"]["state"]["enum"] == ["open", "closed", "all"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful PR listing."""
        tool = GitHubListPullRequestsTool()

        mock_response = [
            {
                "number": 1,
                "title": "Add feature",
                "state": "open",
                "user": {"login": "octocat"},
                "html_url": "https://github.com/octocat/Hello-World/pull/1",
            }
        ]

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                None,
                {"owner": "octocat", "repo": "Hello-World"},
                credentials=make_credentials(),
            )

            assert len(result) == 1
            assert result[0]["number"] == 1

            call_args = mock_request.call_args
            assert "repos/octocat/Hello-World/pulls" in call_args[0][1]


@pytest.mark.unit
class TestGitHubGetPullRequestTool:
    """Tests for GitHubGetPullRequestTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = GitHubGetPullRequestTool()
        assert tool.name == "github_pull_request_read_get"
        assert tool.integration_type == "github"

    @pytest.mark.asyncio
    async def test_execute_missing_pull_number(self):
        """Test execute raises error when pull_number is missing."""
        tool = GitHubGetPullRequestTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                None,
                {"owner": "octocat", "repo": "Hello-World"},
                credentials=make_credentials(),
            )

        assert exc_info.value.status_code == 400
        assert "pullNumber" in exc_info.value.detail or "pull_number" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_accepts_both_param_names(self):
        """Test execute accepts both pullNumber and pull_number."""
        tool = GitHubGetPullRequestTool()

        mock_response = {"number": 123, "title": "Test PR"}

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            # Test with pullNumber
            await tool.execute(
                None,
                {"owner": "octocat", "repo": "Hello-World", "pullNumber": 123},
                credentials=make_credentials(),
            )

            call_args = mock_request.call_args
            assert "pulls/123" in call_args[0][1]

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            # Test with pull_number
            await tool.execute(
                None,
                {"owner": "octocat", "repo": "Hello-World", "pull_number": 456},
                credentials=make_credentials(),
            )

            call_args = mock_request.call_args
            assert "pulls/456" in call_args[0][1]


@pytest.mark.unit
class TestGitHubGetPRFilesTool:
    """Tests for GitHubGetPRFilesTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = GitHubGetPRFilesTool()
        assert tool.name == "github_get_pr_files"
        assert tool.integration_type == "github"
        assert tool.provider == "github"

    def test_get_parameters_schema(self):
        """Test parameter schema includes pagination."""
        tool = GitHubGetPRFilesTool()
        schema = tool.get_parameters_schema()

        assert "pull_number" in schema["properties"]
        assert "per_page" in schema["properties"]
        assert "page" in schema["properties"]
        assert "pull_number" in schema["required"]

    def test_get_output_schema(self):
        """Test output schema describes file array with patches."""
        tool = GitHubGetPRFilesTool()
        schema = tool.get_output_schema()

        assert schema["type"] == "array"
        file_props = schema["items"]["properties"]
        assert "filename" in file_props
        assert "status" in file_props
        assert "patch" in file_props
        assert "additions" in file_props
        assert "deletions" in file_props

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful PR files retrieval."""
        tool = GitHubGetPRFilesTool()

        mock_response = [
            {
                "sha": "abc123",
                "filename": "src/main.py",
                "status": "modified",
                "additions": 10,
                "deletions": 5,
                "changes": 15,
                "patch": "@@ -1,5 +1,10 @@\n+new line",
            },
            {
                "sha": "def456",
                "filename": "README.md",
                "status": "added",
                "additions": 20,
                "deletions": 0,
                "changes": 20,
                "patch": "@@ -0,0 +1,20 @@\n+# Title",
            },
        ]

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                None,
                {"owner": "octocat", "repo": "Hello-World", "pull_number": 123},
                credentials=make_credentials(),
            )

            assert len(result) == 2
            assert result[0]["filename"] == "src/main.py"
            assert result[1]["status"] == "added"
            assert "patch" in result[0]

            call_args = mock_request.call_args
            assert "repos/octocat/Hello-World/pulls/123/files" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_execute_with_pagination(self):
        """Test PR files with pagination parameters."""
        tool = GitHubGetPRFilesTool()

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = []

            await tool.execute(
                None,
                {"owner": "octocat", "repo": "Hello-World", "pull_number": 123, "per_page": 50, "page": 2},
                credentials=make_credentials(),
            )

            call_args = mock_request.call_args
            params = call_args[1].get("params", {})
            assert params["per_page"] == 50
            assert params["page"] == 2


@pytest.mark.unit
class TestGitHubErrorHandling:
    """Tests for GitHub API error handling."""

    @pytest.mark.asyncio
    async def test_auth_failure_401(self):
        """Test 401 error handling for expired/invalid token."""
        tool = GitHubListCommitsTool()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.is_success = False
        mock_response.json.return_value = {"message": "Bad credentials"}
        mock_response.text = "Bad credentials"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            with pytest.raises(HTTPException) as exc_info:
                await tool.execute(
                    None,
                    {"owner": "octocat", "repo": "Hello-World"},
                    credentials=make_credentials(),
                )

            assert exc_info.value.status_code == 401
            assert "authentication" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_not_found_404(self):
        """Test 404 error handling for missing repo/commit."""
        tool = GitHubGetCommitTool()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.is_success = False
        mock_response.json.return_value = {"message": "Not Found"}
        mock_response.text = "Not Found"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            with pytest.raises(HTTPException) as exc_info:
                await tool.execute(
                    None,
                    {"owner": "octocat", "repo": "nonexistent", "sha": "abc123"},
                    credentials=make_credentials(),
                )

            assert exc_info.value.status_code == 404
            assert "not found" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test rate limit error is mapped to 429."""
        tool = GitHubListCommitsTool()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 403
        mock_response.is_success = False
        mock_response.json.return_value = {"message": "API rate limit exceeded"}
        mock_response.text = "API rate limit exceeded"
        mock_response.headers = {"X-RateLimit-Reset": "1705320000"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            with pytest.raises(HTTPException) as exc_info:
                await tool.execute(
                    None,
                    {"owner": "octocat", "repo": "Hello-World"},
                    credentials=make_credentials(),
                )

            assert exc_info.value.status_code == 429
            assert "rate limit" in exc_info.value.detail.lower()
