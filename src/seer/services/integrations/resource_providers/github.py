from __future__ import annotations

import re
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.services.integrations.resource_providers.base import ResourceProvider
from seer.logger import get_logger

logger = get_logger(__name__)


class GitHubResourceProvider(ResourceProvider):
    provider = "github"
    resource_configs: Dict[str, Dict[str, Any]] = {
        "github_repo": {
            "list_endpoint": "https://api.github.com/user/repos",
            "display_field": "full_name",
            "value_field": "full_name",
            "supports_hierarchy": False,
            "supports_search": True,
        },
        "github_branch": {
            "list_endpoint": "https://api.github.com/repos/{owner}/{repo}/branches",
            "display_field": "name",
            "value_field": "name",
            "depends_on": "repo",
            "supports_hierarchy": False,
            "supports_search": False,
        },
    }

    async def list_resources(
        self,
        *,
        access_token: str,
        resource_type: str,
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        filter_params: Optional[Dict[str, Any]],
        depends_on_values: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        if resource_type == "github_repo":
            return await self._list_repos(access_token, query, page_token, page_size)
        if resource_type == "github_branch":
            repo = (depends_on_values or {}).get("repo")
            if not repo:
                return {"items": [], "error": "repo is required"}
            return await self._list_branches(access_token, repo)
        raise HTTPException(status_code=400, detail=f"Unsupported GitHub resource type '{resource_type}'")

    async def _list_repos(
        self,
        access_token: str,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        url = "https://api.github.com/user/repos"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }
        params = {
            "per_page": page_size,
            "sort": "updated",
            "direction": "desc",
        }
        if page_token:
            params["page"] = page_token

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
        except Exception as exc:
            logger.exception("Error listing GitHub repos: %s", exc)
            return {"items": [], "error": str(exc)}

        if response.status_code != 200:
            logger.error("GitHub API error: %s - %s", response.status_code, response.text[:200])
            return {"items": [], "error": f"API error: {response.status_code}"}

        repos = response.json()
        if query:
            query_lower = query.lower()
            repos = [repo for repo in repos if query_lower in (repo.get("full_name") or "").lower()]

        items = []
        for repo in repos:
            items.append(
                {
                    "id": str(repo.get("id")),
                    "name": repo.get("full_name"),
                    "display_name": repo.get("full_name"),
                    "type": "repository",
                    "description": repo.get("description"),
                    "private": repo.get("private"),
                    "web_url": repo.get("html_url"),
                }
            )

        link_header = response.headers.get("Link", "")
        next_page = None
        if 'rel="next"' in link_header:
            match = re.search(r"page=(\\d+)>; rel=\"next\"", link_header)
            if match:
                next_page = match.group(1)

        return {
            "items": items,
            "next_page_token": next_page,
            "supports_hierarchy": False,
            "supports_search": True,
        }

    async def _list_branches(self, access_token: str, repo: str) -> Dict[str, Any]:
        url = f"https://api.github.com/repos/{repo}/branches"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
        except Exception as exc:
            logger.exception("Error listing GitHub branches: %s", exc)
            return {"items": [], "error": str(exc)}

        if response.status_code != 200:
            logger.error("GitHub API error: %s - %s", response.status_code, response.text[:200])
            return {"items": [], "error": f"API error: {response.status_code}"}

        branches = response.json()
        items = []
        for branch in branches:
            items.append(
                {
                    "id": branch.get("name"),
                    "name": branch.get("name"),
                    "display_name": branch.get("name"),
                    "type": "branch",
                    "protected": branch.get("protected", False),
                }
            )

        return {"items": items, "supports_hierarchy": False, "supports_search": False}
