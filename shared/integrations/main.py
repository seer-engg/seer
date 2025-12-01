from .providers.github import GithubProvider
from .providers.asana import AsanaProvider
from .providers.base import BaseProvider

PROVIDERS = {
    "github": GithubProvider,
    "asana": AsanaProvider,
}


async def get_provider(mcp_service: str) -> BaseProvider:
    return PROVIDERS[mcp_service]()


    