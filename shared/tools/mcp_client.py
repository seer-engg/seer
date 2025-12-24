# shared/tools/mcp_client.py

from langchain_mcp_adapters.client import MultiServerMCPClient  
from shared.config import config
import asyncio
from typing import Optional, Any
import threading

from typing import Optional, Any, TypeVar
from typing import Callable, Coroutine
from concurrent.futures import Future

_T = TypeVar("_T")
# Build headers dict, filtering out None values
context7_headers = {}
if config.CONTEXT7_API_KEY:
    context7_headers["CONTEXT7_API_KEY"] = config.CONTEXT7_API_KEY

# Build GitHub MCP server config
github_mcp_config = {}
if config.GITHUB_MCP_SERVER_URL:
    github_mcp_config["github"] = {
        "transport": "streamable_http",
        "url": config.GITHUB_MCP_SERVER_URL,
        "headers": None,  # GitHub MCP server may use OAuth tokens passed per-request
    }

client = MultiServerMCPClient(
    {
        "context7": {
            "transport": "streamable_http",
            "url": "https://mcp.context7.com/mcp",
            "headers": context7_headers if context7_headers else None,
        },
        "langchain": {
            "transport": "streamable_http",
            "url": "https://docs.langchain.com/mcp",
        },
        **github_mcp_config,  # Add GitHub MCP server if configured
    }
)


# LANGCHAIN_DOCS_TOOLS = asyncio.run(client.get_tools(server_name="langchain"))
# CONTEXT7_TOOLS = asyncio.run(client.get_tools(server_name="context7"))


def _run_coro_safely(factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """
    Run the coroutine returned by `factory` even if we're currently inside an event loop.
    When there is no running loop this simply delegates to asyncio.run; otherwise it spins
    up a background thread so we don't trip over "asyncio.run() cannot be called" errors.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    future: Future[_T] = Future()

    def _runner() -> None:
        try:
            result = asyncio.run(factory())
        except BaseException as exc:  # propagate original failure
            future.set_exception(exc)
        else:
            future.set_result(result)

    thread = threading.Thread(
        target=_runner,
        name="mcp-client-loader",
        daemon=True,
    )
    thread.start()
    try:
        return future.result()
    finally:
        thread.join()


def _load_tools(server_name: str):
    return _run_coro_safely(lambda: client.get_tools(server_name=server_name))


# Preload the MCP tools without assuming we control the active event loop
LANGCHAIN_DOCS_TOOLS = _load_tools("langchain")
CONTEXT7_TOOLS = _load_tools("context7")

CONTEXT7_LIBRARY_TOOL = [tool for tool in CONTEXT7_TOOLS if tool.name == "get-library-docs"][0]