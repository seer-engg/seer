from shared.schema import SandboxContext
from indexer.service import index_codebase
from agents.codex.state import CodexState   

async def index(state: CodexState) -> CodexState:
    """
    Index the codebase in the sandbox.
    """
    # TODO: uncomment this when indexer is ready
    return {}

    result = await index_codebase(state.context.sandbox_context)
    return {}