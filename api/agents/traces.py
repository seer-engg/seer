"""
Agent Traces API - List and detail endpoints for agent conversation traces.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel, Field
from shared.logger import get_logger
from .checkpointer import get_checkpointer

logger = get_logger("api.agents.traces")

router = APIRouter(prefix="/agents/traces", tags=["Agent Traces"])


# =============================================================================
# Pydantic Models
# =============================================================================

class AgentMessage(BaseModel):
    """Agent message model matching frontend interface."""
    id: int = Field(..., description="Sequential message ID")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    thinking: Optional[str] = Field(None, description="AI reasoning/thinking")
    suggested_edits: Optional[Dict[str, Any]] = Field(None, description="Suggested edits")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    created_at: str = Field(..., description="ISO timestamp")


class AgentTraceSummary(BaseModel):
    """Agent trace summary for list view."""
    thread_id: str = Field(..., description="Thread ID")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    workflow_name: Optional[str] = Field(None, description="Associated workflow name")
    message_count: int = Field(..., description="Number of messages")
    created_at: str = Field(..., description="ISO timestamp")
    updated_at: str = Field(..., description="ISO timestamp")
    title: Optional[str] = Field(None, description="Trace title")


class AgentTraceListResponse(BaseModel):
    """Response model for trace list endpoint."""
    traces: List[AgentTraceSummary] = Field(..., description="List of traces")
    total: int = Field(..., description="Total number of traces")


class AgentTraceDetail(BaseModel):
    """Agent trace detail with messages."""
    thread_id: str = Field(..., description="Thread ID")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    workflow_name: Optional[str] = Field(None, description="Associated workflow name")
    created_at: str = Field(..., description="ISO timestamp")
    updated_at: str = Field(..., description="ISO timestamp")
    title: Optional[str] = Field(None, description="Trace title")
    messages: List[AgentMessage] = Field(..., description="List of messages")


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_message_to_agent_message(msg: Any, msg_id: int, checkpoint_ts: Optional[str] = None) -> AgentMessage:
    """Convert LangChain message to AgentMessage format."""
    # Handle dict format (from checkpoint)
    if isinstance(msg, dict):
        msg_type = msg.get("type", "") or msg.get("role", "")
        content = msg.get("content", "")
        additional_kwargs = msg.get("additional_kwargs", {})
        metadata = msg.get("metadata", {})
    # Handle LangChain message objects
    elif hasattr(msg, "content") and hasattr(msg, "type"):
        msg_type = msg.type
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        additional_kwargs = getattr(msg, "additional_kwargs", {})
        metadata = getattr(msg, "metadata", {})
    else:
        # Fallback
        msg_type = "unknown"
        content = str(msg)
        additional_kwargs = {}
        metadata = {}
    
    # Determine role
    if msg_type in ("human", "user"):
        role = "user"
    elif msg_type in ("ai", "assistant"):
        role = "assistant"
    else:
        role = "user"  # Default fallback
    
    # Extract thinking/reasoning
    thinking = None
    if isinstance(additional_kwargs, dict):
        thinking = additional_kwargs.get("thinking") or additional_kwargs.get("reasoning")
    if not thinking and isinstance(metadata, dict):
        thinking = metadata.get("thinking") or metadata.get("reasoning")
    
    # Extract suggested_edits
    suggested_edits = None
    if isinstance(additional_kwargs, dict):
        suggested_edits = additional_kwargs.get("suggested_edits") or additional_kwargs.get("suggested_edits")
    if not suggested_edits and isinstance(metadata, dict):
        suggested_edits = metadata.get("suggested_edits")
    
    # Extract created_at
    created_at = checkpoint_ts or datetime.utcnow().isoformat()
    if isinstance(metadata, dict) and "created_at" in metadata:
        created_at = metadata["created_at"]
    
    return AgentMessage(
        id=msg_id,
        role=role,
        content=content,
        thinking=thinking,
        suggested_edits=suggested_edits,
        metadata=metadata if isinstance(metadata, dict) else None,
        created_at=created_at,
    )


def _extract_workflow_id_from_thread_id(thread_id: str) -> Optional[str]:
    """Extract workflow_id from thread_id pattern: workflow-wf_X-{uuid}"""
    if thread_id.startswith("workflow-"):
        # Pattern: workflow-wf_1-{uuid}
        parts = thread_id.split("-", 2)  # Split into ['workflow', 'wf_1', '{uuid}']
        if len(parts) >= 2 and parts[1].startswith("wf_"):
            return parts[1]  # Return 'wf_1'
    return None


async def _get_workflow_name(workflow_id: Optional[str]) -> Optional[str]:
    """Lookup workflow name from database."""
    if not workflow_id:
        return None
    
    try:
        from shared.database.workflow_models import WorkflowRecord
        
        # Parse workflow_id (format: wf_123)
        if workflow_id.startswith("wf_"):
            try:
                workflow_db_id = int(workflow_id[3:])
                workflow = await WorkflowRecord.get_or_none(id=workflow_db_id)
                if workflow:
                    return workflow.name
            except (ValueError, Exception) as e:
                logger.debug(f"Could not lookup workflow name for {workflow_id}: {e}")
    except Exception as e:
        logger.debug(f"Error looking up workflow name: {e}")
    
    return None


def _extract_metadata_from_checkpoint(
    checkpoint: Dict[str, Any],
    channel_values: Dict[str, Any],
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """Extract metadata (workflow_id, title) from checkpoint."""
    metadata = {}
    
    # Check checkpoint metadata
    checkpoint_metadata = checkpoint.get("metadata", {})
    if isinstance(checkpoint_metadata, dict):
        metadata["workflow_id"] = checkpoint_metadata.get("workflow_id")
        metadata["title"] = checkpoint_metadata.get("title")
    
    # Check channel_values for metadata
    if "metadata" in channel_values and isinstance(channel_values["metadata"], dict):
        if not metadata.get("workflow_id"):
            metadata["workflow_id"] = channel_values["metadata"].get("workflow_id")
        if not metadata.get("title"):
            metadata["title"] = channel_values["metadata"].get("title")
    
    # Fallback: extract from thread_id if not found in metadata
    if not metadata.get("workflow_id") and thread_id:
        extracted_id = _extract_workflow_id_from_thread_id(thread_id)
        if extracted_id:
            metadata["workflow_id"] = extracted_id
    
    # Try to extract title from first message if not found
    if not metadata.get("title"):
        messages = channel_values.get("messages", [])
        if messages and isinstance(messages, list) and len(messages) > 0:
            first_msg = messages[0]
            if isinstance(first_msg, dict):
                content = first_msg.get("content", "")
            elif hasattr(first_msg, "content"):
                content = str(first_msg.content)
            else:
                content = str(first_msg)
            
            # Use first 100 chars as title
            if content:
                metadata["title"] = content[:100].strip()
    
    return metadata


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("", response_model=AgentTraceListResponse)
async def list_agent_traces(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> AgentTraceListResponse:
    """
    List all agent traces (excluding workflow executions).
    
    Returns paginated list of agent conversation traces.
    """
    checkpointer = await get_checkpointer()
    if not checkpointer:
        logger.warning("Checkpointer not available, returning empty trace list")
        return AgentTraceListResponse(traces=[], total=0)
    
    try:
        # Collect all checkpoints and group by thread_id
        thread_data: Dict[str, Dict[str, Any]] = {}
        
        async for checkpoint_tuple in checkpointer.alist({}):
            thread_id = checkpoint_tuple.config.get("configurable", {}).get("thread_id")
            if not thread_id:
                continue
            
            # Skip workflow executions (run_* pattern)
            if thread_id.startswith("run_"):
                continue
            
            checkpoint = checkpoint_tuple.checkpoint
            channel_values = checkpoint.get("channel_values", {})
            ts_str = checkpoint.get("ts", "")
            
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except Exception:
                timestamp = datetime.utcnow()
            
            # Initialize thread data if not exists
            if thread_id not in thread_data:
                thread_data[thread_id] = {
                    "thread_id": thread_id,
                    "checkpoints": [],
                    "earliest_ts": timestamp,
                    "latest_ts": timestamp,
                }
            
            # Track earliest and latest timestamps
            thread_data[thread_id]["checkpoints"].append({
                "checkpoint": checkpoint,
                "timestamp": timestamp,
            })
            if timestamp < thread_data[thread_id]["earliest_ts"]:
                thread_data[thread_id]["earliest_ts"] = timestamp
            if timestamp > thread_data[thread_id]["latest_ts"]:
                thread_data[thread_id]["latest_ts"] = timestamp
        
        # Convert to trace summaries
        traces = []
        for thread_id, data in thread_data.items():
            # Get latest checkpoint
            latest_checkpoint = max(data["checkpoints"], key=lambda c: c["timestamp"])["checkpoint"]
            channel_values = latest_checkpoint.get("channel_values", {})
            
            # Extract messages
            messages = channel_values.get("messages", [])
            message_count = len(messages) if isinstance(messages, list) else 0
            
            # Extract metadata
            metadata = _extract_metadata_from_checkpoint(latest_checkpoint, channel_values, thread_id)
            workflow_id = metadata.get("workflow_id")
            title = metadata.get("title")
            
            # Lookup workflow name
            workflow_name = await _get_workflow_name(workflow_id)
            
            trace = AgentTraceSummary(
                thread_id=thread_id,
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                message_count=message_count,
                created_at=data["earliest_ts"].isoformat(),
                updated_at=data["latest_ts"].isoformat(),
                title=title,
            )
            traces.append(trace)
        
        # Sort by updated_at (newest first)
        traces.sort(key=lambda t: t.updated_at, reverse=True)
        
        # Apply pagination
        total = len(traces)
        paginated_traces = traces[offset:offset + limit]
        
        return AgentTraceListResponse(traces=paginated_traces, total=total)
        
    except Exception as e:
        logger.error(f"Error listing agent traces: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list traces: {str(e)}")


@router.get("/{thread_id}", response_model=AgentTraceDetail)
async def get_agent_trace(
    request: Request,
    thread_id: str,
) -> AgentTraceDetail:
    """
    Get detailed agent trace with messages.
    
    Returns full trace detail including all messages.
    """
    checkpointer = await get_checkpointer()
    if not checkpointer:
        raise HTTPException(status_code=503, detail="Checkpointer not available")
    
    try:
        # Get latest checkpoint for thread
        config = {"configurable": {"thread_id": thread_id}}
        state_tuple = await checkpointer.aget_tuple(config)
        
        if not state_tuple:
            raise HTTPException(status_code=404, detail=f"Trace not found for thread_id: {thread_id}")
        
        checkpoint = state_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {})
        ts_str = checkpoint.get("ts", "")
        
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except Exception:
            timestamp = datetime.utcnow()
        
        # Get all checkpoints for this thread to find earliest timestamp
        earliest_ts = timestamp
        async for checkpoint_tuple in checkpointer.alist(config):
            cp_ts_str = checkpoint_tuple.checkpoint.get("ts", "")
            try:
                cp_ts = datetime.fromisoformat(cp_ts_str.replace('Z', '+00:00'))
                if cp_ts < earliest_ts:
                    earliest_ts = cp_ts
            except Exception:
                pass
        
        # Extract messages
        messages_raw = channel_values.get("messages", [])
        messages = []
        for idx, msg in enumerate(messages_raw if isinstance(messages_raw, list) else []):
            agent_msg = _convert_message_to_agent_message(msg, idx, ts_str)
            messages.append(agent_msg)
        
        # Extract metadata
        metadata = _extract_metadata_from_checkpoint(checkpoint, channel_values, thread_id)
        workflow_id = metadata.get("workflow_id")
        title = metadata.get("title")
        
        # Lookup workflow name
        workflow_name = await _get_workflow_name(workflow_id)
        
        return AgentTraceDetail(
            thread_id=thread_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            created_at=earliest_ts.isoformat(),
            updated_at=timestamp.isoformat(),
            title=title,
            messages=messages,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent trace: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get trace: {str(e)}")

