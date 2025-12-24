"""
Traces API router for querying LangGraph checkpoints.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from shared.logger import get_logger
from shared.database.models import User

from .models import TracePublic, TraceDetail, TraceListResponse
from api.agents.checkpointer import get_checkpointer

logger = get_logger("api.traces.router")

router = APIRouter(prefix="/api/traces", tags=["traces"])


def _determine_trace_type(thread_id: str) -> str:
    """Determine trace type from thread_id pattern."""
    # Workflow chat conversations use pattern: workflow-{workflow_id}-{uuid}
    # This pattern has workflow-{number}-{hex_string} format
    # We need to distinguish workflow chat from actual workflow execution traces
    if thread_id.startswith("workflow-"):
        # Check if it matches the workflow chat pattern: workflow-{id}-{uuid}
        # Pattern: workflow-{number}-{hex_string} where hex_string is typically 32 chars
        parts = thread_id.split("-")
        if len(parts) >= 3:
            # Check if third part looks like a UUID hex (typically 32 chars)
            # Workflow chat uses uuid.uuid4().hex which is 32 hex characters
            uuid_part = parts[2]
            if len(uuid_part) == 32 and all(c in '0123456789abcdef' for c in uuid_part.lower()):
                # This is a workflow chat thread
                return "chat"
        # Otherwise, it's a workflow execution trace
        return "workflow"
    elif thread_id.startswith("orchestrator-") or "supervisor" in thread_id.lower():
        return "orchestrator"
    elif "chat" in thread_id.lower() or "thread" in thread_id.lower():
        return "chat"
    else:
        return "unknown"


def _determine_status(checkpoint: Dict[str, Any], has_interrupt: bool = False) -> str:
    """Determine trace status from checkpoint data."""
    if has_interrupt:
        return "interrupted"
    
    # Check for error in channel_values
    channel_values = checkpoint.get("channel_values", {})
    if "error" in channel_values:
        return "failed"
    
    # Check if there's a next node (running) or not (completed)
    # This is a heuristic - we'd need to check the graph state for definitive status
    return "completed"


def _sanitize_checkpoint_data(obj: Any) -> Any:
    """
    Recursively sanitize checkpoint data to remove non-serializable types.
    
    Converts LangGraph-specific types (like Send) to JSON-safe formats.
    """
    if obj is None:
        return None
    
    # Handle primitive types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Handle datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: _sanitize_checkpoint_data(v) for k, v in obj.items()}
    
    # Handle lists
    if isinstance(obj, (list, tuple)):
        return [_sanitize_checkpoint_data(item) for item in obj]
    
    # Handle LangGraph Send type and similar
    if hasattr(obj, "__class__"):
        class_name = obj.__class__.__name__
        module_name = getattr(obj.__class__, "__module__", "")
        
        # Check if it's a LangGraph type
        if "langgraph" in module_name.lower():
            # Try to extract useful information
            if hasattr(obj, "__dict__"):
                return {
                    "type": f"{module_name}.{class_name}",
                    "value": _sanitize_checkpoint_data(obj.__dict__)
                }
            return {
                "type": f"{module_name}.{class_name}",
                "repr": str(obj)
            }
        
        # Handle Pydantic models
        if hasattr(obj, "model_dump"):
            return _sanitize_checkpoint_data(obj.model_dump())
        
        if hasattr(obj, "dict"):
            return _sanitize_checkpoint_data(obj.dict())
    
    # Fallback: convert to string
    return str(obj)


@router.get("", response_model=TraceListResponse)
async def list_traces_endpoint(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    trace_type: Optional[str] = Query(None, description="Filter by trace type: workflow, chat, orchestrator"),
    thread_id_pattern: Optional[str] = Query(None, description="Filter by thread_id pattern"),
) -> TraceListResponse:
    """
    List all traces from Postgres checkpoints.
    
    Queries LangGraph checkpoints to retrieve trace information.
    Supports filtering by trace type and thread_id pattern.
    """
    user: User = request.state.db_user
    
    checkpointer = await get_checkpointer()
    if not checkpointer:
        logger.warning("Checkpointer not available, returning empty trace list")
        return TraceListResponse(traces=[], total=0, limit=limit, offset=offset)
    
    try:
        traces = []
        all_checkpoints = []
        
        # Collect all checkpoints
        # Note: alist() with empty config lists all checkpoints
        # We'll need to filter by thread_id patterns
        async for checkpoint_tuple in checkpointer.alist({}):
            all_checkpoints.append(checkpoint_tuple)
        
        # Group by thread_id and get the latest checkpoint for each thread
        thread_latest: Dict[str, Any] = {}
        thread_checkpoints: Dict[str, List[Any]] = {}
        
        for checkpoint_tuple in all_checkpoints:
            thread_id = checkpoint_tuple.config.get("configurable", {}).get("thread_id")
            if not thread_id:
                continue
            
            # Filter by thread_id pattern if provided
            if thread_id_pattern and thread_id_pattern not in thread_id:
                continue
            
            # Filter by trace type if provided
            trace_type_detected = _determine_trace_type(thread_id)
            logger.debug(f"Trace type detection: thread_id={thread_id}, detected_type={trace_type_detected}")
            if trace_type and trace_type != trace_type_detected:
                continue
            
            # Track latest checkpoint per thread
            if thread_id not in thread_latest:
                thread_latest[thread_id] = checkpoint_tuple
                thread_checkpoints[thread_id] = []
            
            thread_checkpoints[thread_id].append(checkpoint_tuple)
            
            # Update latest if this checkpoint is newer
            current_ts = checkpoint_tuple.checkpoint.get("ts")
            latest_ts = thread_latest[thread_id].checkpoint.get("ts")
            if current_ts and latest_ts:
                try:
                    current_dt = datetime.fromisoformat(current_ts.replace('Z', '+00:00'))
                    latest_dt = datetime.fromisoformat(latest_ts.replace('Z', '+00:00'))
                    if current_dt > latest_dt:
                        thread_latest[thread_id] = checkpoint_tuple
                except Exception:
                    pass
        
        # Convert to TracePublic models
        for thread_id, checkpoint_tuple in thread_latest.items():
            checkpoint = checkpoint_tuple.checkpoint
            checkpoint_id = checkpoint_tuple.config.get("configurable", {}).get("checkpoint_id")
            
            # Extract timestamp
            ts_str = checkpoint.get("ts", "")
            try:
                timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except Exception:
                timestamp = datetime.now()
            
            # Determine trace type
            trace_type_detected = _determine_trace_type(thread_id)
            logger.debug(f"Creating trace: thread_id={thread_id}, trace_type={trace_type_detected}, timestamp={timestamp}")
            
            # Determine status
            # Check if there's an interrupt in the checkpoint
            has_interrupt = checkpoint.get("interrupt") is not None
            status = _determine_status(checkpoint, has_interrupt)
            
            # Extract node name from channel_values
            channel_values = checkpoint.get("channel_values", {})
            node = channel_values.get("node")
            
            # Count messages if available
            messages = channel_values.get("messages", [])
            message_count = len(messages) if isinstance(messages, list) else None
            
            # Calculate duration (if we have multiple checkpoints for this thread)
            duration_ms = None
            if thread_id in thread_checkpoints and len(thread_checkpoints[thread_id]) > 1:
                checkpoints = thread_checkpoints[thread_id]
                try:
                    # Sort checkpoints by timestamp to ensure correct duration calculation
                    sorted_checkpoints = sorted(
                        checkpoints,
                        key=lambda cp: cp.checkpoint.get("ts", ""),
                        reverse=False
                    )
                    first_ts = sorted_checkpoints[0].checkpoint.get("ts")
                    last_ts = sorted_checkpoints[-1].checkpoint.get("ts")
                    if first_ts and last_ts:
                        first_dt = datetime.fromisoformat(first_ts.replace('Z', '+00:00'))
                        last_dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
                        duration_ms = int((last_dt - first_dt).total_seconds() * 1000)
                except Exception:
                    pass
            
            trace = TracePublic(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                timestamp=timestamp,
                trace_type=trace_type_detected,
                status=status,
                node=node,
                metadata={"checkpoint_count": len(thread_checkpoints.get(thread_id, []))},
                message_count=message_count,
                duration_ms=duration_ms,
            )
            traces.append(trace)
        
        # Sort by timestamp (newest first)
        traces.sort(key=lambda t: t.timestamp, reverse=True)
        
        # Apply pagination
        total = len(traces)
        paginated_traces = traces[offset:offset + limit]
        
        return TraceListResponse(
            traces=paginated_traces,
            total=total,
            limit=limit,
            offset=offset,
        )
        
    except Exception as e:
        logger.error(f"Error listing traces: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list traces: {str(e)}"
        )


@router.get("/{thread_id}", response_model=TraceDetail)
async def get_trace_endpoint(
    request: Request,
    thread_id: str,
) -> TraceDetail:
    """
    Get detailed trace information for a specific thread.
    
    Returns all checkpoints for the thread in chronological order.
    """
    user: User = request.state.db_user
    
    checkpointer = await get_checkpointer()
    if not checkpointer:
        raise HTTPException(
            status_code=503,
            detail="Checkpointer not available"
        )
    
    try:
        # Get all checkpoints for this thread
        config = {"configurable": {"thread_id": thread_id}}
        checkpoints = []
        
        async for checkpoint_tuple in checkpointer.alist(config):
            checkpoints.append({
                "checkpoint_id": checkpoint_tuple.config.get("configurable", {}).get("checkpoint_id"),
                "timestamp": checkpoint_tuple.checkpoint.get("ts"),
                "checkpoint": _sanitize_checkpoint_data(checkpoint_tuple.checkpoint),
                "config": _sanitize_checkpoint_data(checkpoint_tuple.config),
            })
        
        if not checkpoints:
            raise HTTPException(
                status_code=404,
                detail=f"Trace not found for thread_id: {thread_id}"
            )
        
        # Sort by timestamp
        checkpoints.sort(key=lambda c: c.get("timestamp", ""))
        
        # Get current state
        current_state = None
        try:
            state_tuple = await checkpointer.aget_tuple(config)
            if state_tuple:
                current_state = {
                    "values": _sanitize_checkpoint_data(state_tuple.checkpoint.get("channel_values", {})),
                    "next": _sanitize_checkpoint_data(getattr(state_tuple, "next", None)),
                }
        except Exception:
            pass
        
        return TraceDetail(
            thread_id=thread_id,
            checkpoints=checkpoints,
            current_state=current_state,
            metadata={"checkpoint_count": len(checkpoints)},
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trace: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trace: {str(e)}"
        )


__all__ = ["router"]
