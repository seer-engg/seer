from .providers.github import GithubProvider
from .providers.asana import AsanaProvider
from .providers.base import BaseProvider
from shared.tools.postgres import PostgresProvider

PROVIDERS = {
    "github": GithubProvider,
    "asana": AsanaProvider,
    "postgres": PostgresProvider,
}


async def get_provider(mcp_service: str) -> BaseProvider:
    return PROVIDERS[mcp_service]()


    