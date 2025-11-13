import re
from typing import Optional, Tuple, List 
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agents.eval_agent.constants import LLM
from agents.eval_agent.models import EvalAgentState
from shared.schema import GithubContext, UserContext
from shared.logger import get_logger


logger = get_logger("eval_agent.plan")

def _parse_github_url(url: str, branch_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Parse a GitHub URL and extract the repository URL and branch name.
    
    Handles both:
    - Web URLs: https://github.com/owner/repo/tree/branch-name
    - Git URLs: https://github.com/owner/repo or https://github.com/owner/repo.git
    
    Args:
        url: The GitHub URL (can be a web URL with /tree/ or a git URL)
        branch_name: Optional branch name to use if not in URL
    
    Returns:
        Tuple of (repo_url, branch_name)
    """
    # Pattern to match GitHub web URLs with /tree/ path
    # The branch name can contain slashes, so we match everything after /tree/ 
    # up to an optional trailing slash or path
    web_url_pattern = r'^(https?://github\.com/[^/]+/[^/]+)/tree/([^/]+(?:/[^/]+)*)/?(?:/.+)?$'
    match = re.match(web_url_pattern, url)
    
    if match:
        # Extract repo URL and branch from web URL
        repo_url = match.group(1)
        extracted_branch = match.group(2)
        logger.info(f"Parsed GitHub web URL: repo_url={repo_url}, branch={extracted_branch}")
        return repo_url, extracted_branch
    
    # If it's a standard git URL, use it as-is
    # Remove trailing .git if present for consistency
    repo_url = re.sub(r'\.git$', '', url)
    final_branch = branch_name or "main"
    
    return repo_url, final_branch




async def ensure_target_agent_config(state: EvalAgentState) -> dict:
    last_human = None
    for msg in reversed(state.messages or []):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            last_human = msg
            break
    if last_human is None:
        raise ValueError("No human message to extract from")

    instruction = (
        "Extract the following fields from the user's latest message about the target agent:\n"
        "- github_context: the GitHub context for the target agent\n"
        "- user_context: the user context for the target agent\n"
        "- mcp_services: A list of external service names mentioned (e.g., 'asana', 'github', 'jira'). Return an empty list if none are mentioned.\n"
    )

    class TargetAgentExtractionContext(BaseModel):
        """Context for extracting the target agent's GitHub and user context."""
        github_context: GithubContext
        user_context: UserContext
        mcp_services: List[str] = Field(
            default_factory=list, 
            description="List of external MCP services mentioned, e.g., ['asana', 'github']"
        )

    extractor = LLM.with_structured_output(TargetAgentExtractionContext)
    context: TargetAgentExtractionContext = await extractor.ainvoke(f"{instruction}\n\nUSER:\n{last_human.content}")
    context.user_context.raw_request = last_human.content
    
    # Normalize the GitHub URL in case it's a web URL with /tree/ in it
    normalized_repo_url, normalized_branch = _parse_github_url(
        context.github_context.repo_url, 
        context.github_context.branch_name
    )
    context.github_context.repo_url = normalized_repo_url
    context.github_context.branch_name = normalized_branch
    
    logger.info(f"Extracted required MCP services: {context.mcp_services}")

    return {
        "github_context": context.github_context,
        "user_context": context.user_context,
        "mcp_services": context.mcp_services,
    }
