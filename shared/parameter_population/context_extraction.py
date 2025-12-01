"""
Layer 1: Context Variable Extraction

Extracts structured variables from all available context sources:
- GitHub context (repo URL → owner/repo)
- User context (raw request parsing)
- MCP resources (provisioned resource IDs)
- Environment variables (service defaults)

Design: Service-agnostic extraction with clear naming conventions.
"""
import os
import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from shared.schema import UserContext, GithubContext
from shared.logger import get_logger
from shared.config import config

logger = get_logger("parameter_population.context_extraction")


def extract_all_context_variables(
    user_context: Optional[UserContext] = None,
    github_context: Optional[GithubContext] = None,
    mcp_resources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract all structured variables from available contexts.
    
    Returns a flat dict with clear naming:
    {
        "github_owner": "seer-engg",
        "github_repo": "buggy-coder",
        "github_branch": "main",
        "asana_workspace_gid": "123...",
        "asana_project_gid": "456...",
        ...
    }
    
    Args:
        user_context: User preferences and raw request
        github_context: GitHub repository information
        mcp_resources: Provisioned MCP resources
        
    Returns:
        Flat dict of all available context variables
    """
    context_vars = {}
    
    # Extract GitHub variables
    if github_context:
        github_vars = _extract_github_variables(github_context)
        context_vars.update(github_vars)
    
    # Extract MCP resource variables
    if mcp_resources:
        resource_vars = _extract_mcp_resource_variables(mcp_resources)
        context_vars.update(resource_vars)
    
    # Extract environment variable defaults
    env_vars = _extract_environment_defaults()
    # Env vars are lowest priority - don't overwrite existing
    for key, value in env_vars.items():
        if key not in context_vars:
            context_vars[key] = value
    
    logger.debug(f"Extracted context variables: {list(context_vars.keys())}")
    return context_vars


def _extract_github_variables(github_context: GithubContext) -> Dict[str, Any]:
    """Extract GitHub owner, repo, branch from GithubContext."""
    variables = {}
    
    if github_context.repo_url:
        owner, repo = _parse_github_url(github_context.repo_url)
        if owner:
            variables["github_owner"] = owner
        if repo:
            variables["github_repo"] = repo
    
    if github_context.branch_name:
        variables["github_branch"] = github_context.branch_name
    
    if github_context.agent_name:
        variables["github_agent_name"] = github_context.agent_name
    
    return variables


def _parse_github_url(url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse GitHub URL into (owner, repo).
    
    Handles:
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - https://github.com/owner/repo/tree/branch
    - git@github.com:owner/repo.git
    
    Returns:
        (owner, repo) or (None, None) if parsing fails
    """
    try:
        # Handle git@ URLs
        if url.startswith("git@"):
            # git@github.com:owner/repo.git → owner/repo.git
            match = re.search(r'git@[^:]+:(.+)', url)
            if match:
                path = match.group(1)
            else:
                return None, None
        else:
            # Handle https URLs
            parsed = urlparse(url)
            path = parsed.path.lstrip('/')
        
        # Split path and extract owner/repo
        parts = path.split('/')
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1]
            
            # Clean up repo name
            repo = repo.replace('.git', '')
            
            return owner, repo
    
    except Exception as e:
        logger.warning(f"Failed to parse GitHub URL '{url}': {e}")
    
    return None, None


def _extract_mcp_resource_variables(mcp_resources: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract variables from MCP resources.
    
    For each resource, extract:
    - {resource_name}_id: primary identifier
    - {resource_name}_gid: Asana-style GID
    - {resource_name}_name: display name
    - {resource_name}_type: resource type
    
    Also infers service-level defaults:
    - asana_workspace_gid: first workspace found
    - asana_project_gid: first project found
    """
    variables = {}
    
    # Service-level defaults (first of each type)
    first_workspace_gid = None
    first_project_gid = None
    
    for resource_name, resource_data in mcp_resources.items():
        if not isinstance(resource_data, dict):
            continue
        
        # Handle nested 'data' key (common pattern)
        data = resource_data.get("data", resource_data)
        if not isinstance(data, dict):
            continue
        
        # Extract primary ID
        resource_id = data.get("id") or data.get("gid")
        if resource_id:
            variables[f"{resource_name}_id"] = resource_id
        
        # Extract GID (Asana-specific)
        gid = data.get("gid")
        if gid:
            variables[f"{resource_name}_gid"] = gid
        
        # Extract name
        name = data.get("name") or data.get("full_name")
        if name:
            variables[f"{resource_name}_name"] = name
        
        # Extract type
        resource_type = data.get("type") or data.get("resource_type")
        if resource_type:
            variables[f"{resource_name}_type"] = resource_type
            
            # Track first of each type for service defaults
            if resource_type.lower() == "workspace" and not first_workspace_gid:
                first_workspace_gid = gid or resource_id
            elif resource_type.lower() == "project" and not first_project_gid:
                first_project_gid = gid or resource_id
    
    # Set service-level defaults
    if first_workspace_gid:
        variables["asana_workspace_gid"] = first_workspace_gid
    if first_project_gid:
        variables["asana_project_gid"] = first_project_gid
    
    return variables


def _extract_environment_defaults() -> Dict[str, Any]:
    """
    Extract default values from environment variables.
    
    Returns:
        Dict of environment-based defaults
    """
    variables = {}
    
    # Asana defaults
    asana_workspace = (
        config.asana_workspace_id 
    )
    if asana_workspace:
        variables["asana_workspace_gid"] = asana_workspace.strip()
    
    # GitHub defaults (less common, but possible)
    github_owner = config.github_default_owner
    if github_owner:
        variables["github_owner"] = github_owner.strip()
    
    github_repo = config.github_default_repo
    if github_repo:
        variables["github_repo"] = github_repo.strip()
    
    return variables
