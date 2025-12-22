from typing import List, Optional

from fastapi import HTTPException

from api.traces.models import TraceDetail, TraceSummary


async def list_traces(
    project_name: Optional[str],
    limit: int,
    start_time: Optional[str],
) -> List[TraceSummary]:
    """List traces from MLflow. Langfuse support has been removed."""
    # TODO: Implement MLflow trace listing
    raise HTTPException(
        status_code=501,
        detail="Trace listing is not yet implemented for MLflow. Langfuse support has been removed.",
    )


async def get_trace_detail(
    trace_id: str,
    project_name: Optional[str],
) -> TraceDetail:
    """Get full trace details including nested observations. Langfuse support has been removed."""
    # TODO: Implement MLflow trace detail retrieval
    raise HTTPException(
        status_code=501,
        detail="Trace detail retrieval is not yet implemented for MLflow. Langfuse support has been removed.",
    )


__all__ = ["list_traces", "get_trace_detail"]
