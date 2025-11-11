from shared.schema import SandboxContext
from indexer.service import index_codebase
from agents.codex.state import CodexState   

async def index(state: CodexState) -> CodexState:
    """
    Index the codebase in the sandbox.
    """
    result = await index_codebase(state.updated_sandbox_context)
    return {
        "index_result": result,
    }