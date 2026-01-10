"""
Bootstrap router for consolidated data fetching.

Provides a single endpoint that returns data from multiple sources in parallel.
"""
from fastapi import APIRouter, Request

from shared.database import User

from .schema import BootstrapResponse
from .services import fetch_bootstrap_data

router = APIRouter(prefix="/bootstrap", tags=["bootstrap"])


@router.get("", response_model=BootstrapResponse)
async def get_bootstrap_data(request: Request) -> BootstrapResponse:
    """
    Get all bootstrap data in a single request.

    Consolidates these endpoints into one:
    - /api/tools (45 tools)
    - /api/models (LLM models)
    - /api/integrations/tools/status (connection status)
    - /api/integrations/ (connected accounts)
    - /api/v1/builder/node-types (workflow blocks)
    - /api/v1/workflows (user workflows)

    All data is fetched in parallel using asyncio.gather, so response time
    equals the slowest query (not the sum). Failed sections return empty
    arrays/dicts without causing a 500 error.

    Expected response time: 2-3 seconds (vs 16 seconds for sequential calls)
    Expected response size: ~44 KB uncompressed (~8-10 KB gzipped)
    """
    user: User = request.state.db_user
    data = await fetch_bootstrap_data(user)
    return BootstrapResponse(**data)
