from typing import List, Optional

from fastapi import APIRouter, Query

from api.traces.models import TraceDetail, TraceSummary
from api.traces.services import get_trace_detail, list_traces


router = APIRouter(prefix="/api/traces", tags=["traces"])


@router.get("", response_model=List[TraceSummary])
async def list_traces_endpoint(
    project_name: Optional[str] = Query(
        None,
        description=(
            "Filter by project_name metadata "
            "(e.g., 'eval-v1', 'supervisor-v1', 'codex-v1')"
        ),
    ),
    limit: int = Query(50, ge=1, le=200),
    start_time: Optional[str] = Query(
        None,
        description="ISO 8601 datetime string",
    ),
) -> List[TraceSummary]:
    """
    List traces from Langfuse, optionally filtered by project_name metadata.
    """
    return await list_traces(project_name=project_name, limit=limit, start_time=start_time)


@router.get("/{trace_id}", response_model=TraceDetail)
async def get_trace_detail_endpoint(
    trace_id: str,
    project_name: Optional[str] = Query(
        None,
        description="Optional: Filter by project_name metadata",
    ),
) -> TraceDetail:
    """
    Get full trace details.
    Optionally filters by project_name metadata to ensure trace belongs to the project.
    """
    return await get_trace_detail(trace_id=trace_id, project_name=project_name)


__all__ = ["router"]


