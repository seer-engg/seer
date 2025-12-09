import re
from typing import Optional, Tuple, List 
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from agents.eval_agent.models import EvalAgentPlannerState
from shared.schema import AgentContext
from shared.schema import GithubContext, UserContext
from shared.logger import get_logger
from shared.tools import resolve_mcp_services
from shared.config import config
from langchain_openai import ChatOpenAI


logger = get_logger("eval_agent.plan")

def _is_valid_github_url(url: str) -> bool:
    """
    Validate that a string is a valid GitHub URL format.
    
    Args:
        url: The string to validate
    
    Returns:
        True if it looks like a valid GitHub URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    
    # Check for error indicators
    error_indicators = [
        "Error requesting user input",
        "Error:",
        "Exception:",
        "Traceback",
        "Interrupt(value=",
        "fatal:",
    ]
    url_lower = url.lower()
    for indicator in error_indicators:
        if indicator.lower() in url_lower:
            return False
    
    # Check for valid GitHub URL patterns
    github_patterns = [
        r'^https?://github\.com/[^/]+/[^/]+',  # Basic GitHub URL
        r'^git@github\.com:[^/]+/[^/]+',  # SSH format
    ]
    
    for pattern in github_patterns:
        if re.match(pattern, url):
            return True
    
    return False


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
    
    Raises:
        ValueError: If the URL is not a valid GitHub URL format
    """
    # Validate URL first
    if not _is_valid_github_url(url):
        raise ValueError(f"Invalid GitHub URL format: '{url[:100]}...' (appears to be an error message or invalid format)")
    
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




async def ensure_target_agent_config(state: EvalAgentPlannerState) -> dict:
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
        github_context: Optional[GithubContext] = Field(
            default=None,
            description="GitHub context if a repository URL is mentioned. Can be None in plan-only mode."
        )
        user_context: UserContext
        mcp_services: List[str] = Field(
            default_factory=list, 
            description="List of external MCP services mentioned, e.g., ['asana', 'github']"
        )
        agent_name: str = Field(
            default="agent",
            description="The name of the agent"
        )

    # using gpt-5-mini without response api  to avoid json schema error 
    # TODO: find out how to adapt schea with responses api
    extractor = ChatOpenAI(model="gpt-5-mini", temperature=0.0).with_structured_output(TargetAgentExtractionContext)
    context: TargetAgentExtractionContext = await extractor.ainvoke(f"{instruction}\n\nUSER:\n{last_human.content}")
    context.user_context.raw_request = last_human.content
    
    # Handle GitHub context - normalize if present, create default if missing (for plan-only mode)
    if context.github_context and context.github_context.repo_url and context.github_context.repo_url.strip():
        # Normalize the GitHub URL in case it's a web URL with /tree/ in it
        normalized_repo_url, normalized_branch = _parse_github_url(
            context.github_context.repo_url, 
            context.github_context.branch_name
        )
        context.github_context.repo_url = normalized_repo_url
        context.github_context.branch_name = normalized_branch
    elif not context.github_context or not context.github_context.repo_url or not context.github_context.repo_url.strip():
        # No GitHub URL found
        if config.eval_plan_only_mode:
            # In plan-only mode, GitHub context is optional - create a default one
            logger.info("plan.ensure_config: No GitHub URL found, but plan-only mode - creating default GitHub context")
            context.github_context = GithubContext(
                repo_url="",  # Empty - will be skipped in provision_target_agent
                branch_name="main"
            )
        else:
            # In execution mode, GitHub context is required - raise error asking user to provide it
            raise ValueError(
                "GitHub repository URL is required to evaluate your agent. "
                "Please provide a GitHub repository URL in your next message. "
                "Example: https://github.com/owner/repo"
            )
    
    resolved_services = resolve_mcp_services(context.mcp_services)
    logger.info(
        f"Resolved MCP services (requested={context.mcp_services}): {resolved_services}"
    )
    
    # Create or update the AgentContext
    agent_context = state.context if state.context else AgentContext()
    
    # Update the context with extracted values
    # Ensure github_context is set (should be set above, but handle None case)
    github_ctx = context.github_context
    if not github_ctx:
        if config.eval_plan_only_mode:
            # Plan-only mode: create empty GitHub context
            github_ctx = GithubContext(repo_url="", branch_name="main")
        else:
            raise ValueError("GitHub context is required for execution mode")
    
    updated_context = AgentContext(
        user_context=context.user_context,
        github_context=github_ctx,
        sandbox_context=agent_context.sandbox_context,  # Preserve existing sandbox
        target_agent_version=agent_context.target_agent_version,
        mcp_services=resolved_services,
        mcp_resources=agent_context.mcp_resources,
        agent_name=context.agent_name,
        integrations=agent_context.integrations,
    )

    return {
        "context": updated_context,
    }
