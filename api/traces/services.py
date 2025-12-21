import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from src.core.config import LANGFUSE_BASE_URL, get_auth_headers
from src.traces.models import RunNode, TraceDetail, TraceSummary


async def list_traces(
    project_name: Optional[str],
    limit: int,
    start_time: Optional[str],
) -> List[TraceSummary]:
    """List traces from Langfuse, optionally filtered by project_name and start_time."""
    headers = get_auth_headers()

    # Parse start_time if provided
    start_datetime: Optional[datetime] = None
    if start_time:
        try:
            start_datetime = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid start_time format. Use ISO 8601 format.",
            )

    try:
        # Build filter array for Langfuse API
        filter_array: List[Dict[str, Any]] = []

        # Add metadata filter if project_name is specified
        if project_name:
            filter_array.append(
                {
                    "type": "stringObject",
                    "column": "metadata",
                    "key": "project_name",
                    "operator": "=",
                    "value": project_name,
                }
            )

        # Add timestamp filter if start_time is provided
        if start_datetime:
            filter_array.append(
                {
                    "type": "datetime",
                    "column": "timestamp",
                    "operator": ">=",
                    "value": start_datetime.isoformat(),
                }
            )

        params: Dict[str, Any] = {
            "limit": limit,
            "page": 1,
        }

        # Add filter parameter if filters are specified
        if filter_array:
            params["filter"] = json.dumps(filter_array)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{LANGFUSE_BASE_URL}/api/public/traces",
                headers=headers,
                params=params,
                timeout=30.0,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error fetching traces from Langfuse: {response.text[:500]}",
                )

            data = response.json()
            # Langfuse API returns traces in "data" field, not "traces"
            traces_data = data.get("data", data.get("traces", []))

        # Format traces
        all_traces: List[TraceSummary] = []
        for trace_data in traces_data:
            start_time_obj = (
                datetime.fromisoformat(trace_data["startTime"].replace("Z", "+00:00"))
                if trace_data.get("startTime")
                else None
            )
            end_time_obj = (
                datetime.fromisoformat(trace_data["endTime"].replace("Z", "+00:00"))
                if trace_data.get("endTime")
                else None
            )

            # Extract project_name from metadata, fallback to "unknown"
            metadata = trace_data.get("metadata", {})
            trace_project = (
                metadata.get("project_name", "unknown")
                if isinstance(metadata, dict)
                else "unknown"
            )

            trace = TraceSummary(
                id=trace_data.get("id", ""),
                name=trace_data.get("name", "Unnamed"),
                project=trace_project,
                start_time=start_time_obj,
                end_time=end_time_obj,
                duration=(
                    (end_time_obj - start_time_obj).total_seconds()
                    if (start_time_obj and end_time_obj)
                    else None
                ),
                status=trace_data.get("status", "SUCCESS").lower()
                if trace_data.get("status")
                else "success",
                error=trace_data.get("error")
                if trace_data.get("status") == "ERROR"
                else None,
                run_type="trace",
                inputs=trace_data.get("input"),
                outputs=trace_data.get("output"),
            )
            all_traces.append(trace)

        # Sort by start_time descending
        all_traces.sort(
            key=lambda x: x.start_time or datetime.min,
            reverse=True,
        )

        return all_traces[:limit]

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching traces: {str(exc)}",
        ) from exc


async def get_trace_detail(
    trace_id: str,
    project_name: Optional[str],
) -> TraceDetail:
    """Get full trace details including nested observations."""
    headers = get_auth_headers()

    try:
        async with httpx.AsyncClient() as client:
            # Fetch trace
            response = await client.get(
                f"{LANGFUSE_BASE_URL}/api/public/traces/{trace_id}",
                headers=headers,
                timeout=30.0,
            )

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Trace not found")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error fetching trace: {response.text[:500]}",
                )

            trace_data = response.json()

            # If project_name filter is specified, verify trace metadata matches
            if project_name:
                metadata = trace_data.get("metadata", {})
                trace_project_name = (
                    metadata.get("project_name") if isinstance(metadata, dict) else None
                )
                if trace_project_name != project_name:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Trace not found for project '{project_name}'",
                    )

            # Fetch observations
            obs_response = await client.get(
                f"{LANGFUSE_BASE_URL}/api/public/traces/{trace_id}/observations",
                headers=headers,
                params={"limit": 1000},
                timeout=30.0,
            )

            observations: List[Dict[str, Any]] = []
            if obs_response.status_code == 200:
                obs_data = obs_response.json()
                observations = obs_data.get("observations", [])

        # Format as TraceDetail
        start_time_obj = (
            datetime.fromisoformat(trace_data["startTime"].replace("Z", "+00:00"))
            if trace_data.get("startTime")
            else None
        )
        end_time_obj = (
            datetime.fromisoformat(trace_data["endTime"].replace("Z", "+00:00"))
            if trace_data.get("endTime")
            else None
        )

        # Extract project_name from metadata
        metadata = trace_data.get("metadata", {})
        trace_project = (
            metadata.get("project_name", "unknown")
            if isinstance(metadata, dict)
            else "unknown"
        )

        # Build children tree from observations
        children_map: Dict[str, List[Dict[str, Any]]] = {}
        root_observation: Optional[Dict[str, Any]] = None

        for obs in observations:
            parent_id = obs.get("parentObservationId")

            if not parent_id:
                if not root_observation:
                    root_observation = obs
            else:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(obs)

        def format_obs_node(obs: Dict[str, Any]) -> RunNode:
            obs_id = obs.get("id", "")
            children_obs = children_map.get(obs_id, [])

            start_time_obj = (
                datetime.fromisoformat(obs["startTime"].replace("Z", "+00:00"))
                if obs.get("startTime")
                else None
            )
            end_time_obj = (
                datetime.fromisoformat(obs["endTime"].replace("Z", "+00:00"))
                if obs.get("endTime")
                else None
            )

            return RunNode(
                id=obs_id,
                name=obs.get("name", "Unnamed"),
                run_type=obs.get("type", "unknown").lower(),
                start_time=start_time_obj,
                end_time=end_time_obj,
                duration=(
                    (end_time_obj - start_time_obj).total_seconds()
                    if (start_time_obj and end_time_obj)
                    else None
                ),
                status=obs.get("status", "SUCCESS").lower()
                if obs.get("status")
                else "success",
                error=obs.get("error") if obs.get("status") == "ERROR" else None,
                inputs=obs.get("input"),
                outputs=obs.get("output"),
                children=[
                    format_obs_node(child)
                    for child in sorted(
                        children_obs,
                        key=lambda x: datetime.fromisoformat(
                            x.get("startTime", "").replace("Z", "+00:00")
                        )
                        if x.get("startTime")
                        else datetime.min,
                    )
                ],
            )

        root_children = children_map.get(
            root_observation.get("id", "") if root_observation else "",
            [],
        )

        return TraceDetail(
            id=trace_data.get("id", trace_id),
            name=trace_data.get("name", "Unnamed"),
            project=trace_project,
            start_time=start_time_obj,
            end_time=end_time_obj,
            duration=(
                (end_time_obj - start_time_obj).total_seconds()
                if (start_time_obj and end_time_obj)
                else None
            ),
            status=trace_data.get("status", "SUCCESS").lower()
            if trace_data.get("status")
            else "success",
            error=trace_data.get("error")
            if trace_data.get("status") == "ERROR"
            else None,
            run_type="trace",
            inputs=trace_data.get("input"),
            outputs=trace_data.get("output"),
            children=[
                format_obs_node(child)
                for child in sorted(
                    root_children,
                    key=lambda x: datetime.fromisoformat(
                        x.get("startTime", "").replace("Z", "+00:00")
                    )
                    if x.get("startTime")
                    else datetime.min,
                )
            ],
        )

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching trace: {str(exc)}",
        ) from exc


__all__ = ["list_traces", "get_trace_detail"]


