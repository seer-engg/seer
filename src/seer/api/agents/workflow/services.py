"""
Workflow service layer for the workflow agent APIs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from seer.database import User
from seer.database import (
    Workflow,
    WorkflowChatMessage,
    WorkflowChatSession,
    WorkflowDraft,
    WorkflowProposal,
    parse_workflow_public_id,
)
from seer.logger import get_logger

logger = get_logger("api.workflow_agent.services")


# ============================================================================
# Workflow helpers
# ============================================================================

def _workflow_state_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert compiler WorkflowSpec JSON into a lightweight graph snapshot."""
    nodes_payload: List[Dict[str, Any]] = []
    spec_nodes = spec.get("nodes")
    if isinstance(spec_nodes, list):
        for raw_node in spec_nodes:
            if not isinstance(raw_node, dict):
                continue
            meta = raw_node.get("meta") if isinstance(raw_node.get("meta"), dict) else {}
            label = meta.get("label") if meta else None
            position = meta.get("position") if meta else None

            node_state: Dict[str, Any] = {
                "id": raw_node.get("id"),
                "type": raw_node.get("type"),
                "data": {
                    "label": label or raw_node.get("id"),
                    "config": raw_node,
                },
            }
            if isinstance(position, dict):
                node_state["position"] = {
                    "x": position.get("x", 0),
                    "y": position.get("y", 0),
                }
            nodes_payload.append(node_state)

    edges_payload: List[Dict[str, Any]] = []
    for idx in range(len(nodes_payload) - 1):
        source = nodes_payload[idx].get("id")
        target = nodes_payload[idx + 1].get("id")
        if source and target:
            edges_payload.append(
                {
                    "id": f"wf_edge_{idx}",
                    "source": source,
                    "target": target,
                }
            )
    return {"nodes": nodes_payload, "edges": edges_payload}


def workflow_state_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Public helper to build a workflow-state snapshot from a WorkflowSpec payload."""
    if not isinstance(spec, dict):
        return {"nodes": [], "edges": []}
    return _workflow_state_from_spec(spec)


def workflow_state_snapshot(workflow: Workflow) -> Dict[str, Any]:
    """Return the workflow draft's latest state in ReactFlow-friendly format."""
    draft: Optional[WorkflowDraft] = getattr(workflow, "draft", None)
    if draft and isinstance(draft.spec, dict):
        return workflow_state_from_spec(draft.spec)
    return {"nodes": [], "edges": []}


async def _ensure_workflow_draft(workflow: Workflow) -> WorkflowDraft:
    """
    Ensure we return a resolved WorkflowDraft instance.

    Tortoise attaches a relation manager to unfetched reverse one-to-one fields,
    which is truthy but not the actual model instance. Explicitly check for that
    to avoid accidentally returning the manager and blowing up downstream when
    we try to mutate attributes such as `spec`.
    """

    draft_attr = getattr(workflow, "draft", None)
    draft: Optional[WorkflowDraft] = (
        draft_attr if isinstance(draft_attr, WorkflowDraft) else None
    )

    if draft is None:
        draft = await WorkflowDraft.get_or_none(workflow=workflow)

    if draft is None:
        raise HTTPException(
            status_code=500,
            detail="Workflow draft state not initialized",
        )

    # Cache the resolved draft on the workflow instance for future callers.
    workflow.draft = draft
    return draft


async def _get_workflow(user: User, workflow_id: str) -> Workflow:
    """Resolve and authorize workflow by public id."""
    try:
        pk = parse_workflow_public_id(workflow_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid workflow id format") from exc

    workflow = (
        await Workflow.filter(id=pk, user=user)
        .prefetch_related("draft")
        .first()
    )
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    await workflow.fetch_related("draft")
    return workflow


async def get_workflow(user: User, workflow_id: str) -> Workflow:
    """Public accessor used by routers."""
    return await _get_workflow(user, workflow_id)


# ============================================================================
# Chat Session Services
# ============================================================================

async def create_chat_session(
    workflow: Workflow,
    user: User,
    thread_id: str,
    title: Optional[str] = None,
) -> WorkflowChatSession:
    """
    Create a new chat session for a workflow.
    """
    session = await WorkflowChatSession.create(
        workflow=workflow,
        user=user,
        thread_id=thread_id,
        title=title,
    )

    await session.fetch_related("user")

    logger.info(f"Created chat session {session.id} for workflow {workflow.workflow_id}")
    return session


async def get_chat_session(
    session_id: int,
    workflow: Workflow,
) -> WorkflowChatSession:
    """
    Get a chat session with its messages.

    Args:
        session_id: Session ID
        workflow_id: Workflow ID (for authorization)

    Returns:
        Chat session with messages

    Raises:
        HTTPException: If session not found or unauthorized
    """
    session = await WorkflowChatSession.filter(
        id=session_id,
        workflow=workflow,
    ).prefetch_related('user').first()

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Chat session {session_id} not found"
        )

    return session


async def get_chat_session_by_thread_id(
    thread_id: str,
    workflow: Workflow,
) -> Optional[WorkflowChatSession]:
    """
    Get a chat session by thread ID.

    Args:
        thread_id: LangGraph thread ID
        workflow_id: Workflow ID (for authorization)
        user_id: User ID for authorization (None in self-hosted mode)

    Returns:
        Chat session if found, None otherwise
    """
    session = await WorkflowChatSession.filter(
        thread_id=thread_id,
        workflow=workflow,
    ).prefetch_related('user').first()

    if not session:
        return None

    return session


async def list_chat_sessions(
    workflow: Workflow,
    user: User,
    limit: int = 50,
    offset: int = 0,
) -> List[WorkflowChatSession]:
    """
    List chat sessions for a workflow.

    Args:
        workflow_id: Workflow ID
        user: User
        limit: Maximum number of sessions to return
        offset: Number of sessions to skip

    Returns:
        List of chat sessions
    """
    sessions = await WorkflowChatSession.filter(
        workflow=workflow,
        user=user,
    ).prefetch_related('user').order_by('-created_at').offset(offset).limit(limit).all()

    return sessions


async def save_chat_message(
    session_id: int,
    role: str,
    content: str,
    thinking: Optional[str] = None,
    suggested_edits: Optional[dict] = None,
    metadata: Optional[dict] = None,
    proposal: Optional[WorkflowProposal] = None,
) -> WorkflowChatMessage:
    """
    Save a chat message to the database.

    Args:
        session_id: Session ID
        role: Message role ('user' or 'assistant')
        content: Message content
        thinking: Optional thinking/reasoning steps
        suggested_edits: Optional suggested workflow edits
        metadata: Optional metadata (model used, etc.)
        proposal: Optional proposal linked to this message

    Returns:
        Created message
    """
    session = await WorkflowChatSession.get_or_none(id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Update session updated_at timestamp
    session.updated_at = datetime.utcnow()
    await session.save()

    message = await WorkflowChatMessage.create(
        session=session,
        proposal=proposal,
        role=role,
        content=content,
        thinking=thinking,
        suggested_edits=suggested_edits,
        metadata=metadata,
    )

    logger.debug(f"Saved chat message {message.id} to session {session_id}")
    return message


async def load_chat_history(
    session_id: int,
) -> List[WorkflowChatMessage]:
    """
    Load chat history for a session.

    Args:
        session_id: Session ID

    Returns:
        List of messages ordered by creation time
    """
    messages = await WorkflowChatMessage.filter(
        session_id=session_id
    ).prefetch_related('proposal__created_by', 'proposal__workflow', 'proposal__session').order_by('created_at').all()

    return messages


def _normalize_spec(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and normalize a WorkflowSpec payload."""
    if not spec:
        raise HTTPException(status_code=400, detail="Workflow spec is required")
    try:
        # Lazy import to avoid circular deps
        from seer.core.compiler.parse import parse_workflow_spec

        validated = parse_workflow_spec(spec)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid workflow spec: {exc}") from exc
    return validated.model_dump(mode="json")


def _preview_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build a lightweight preview graph UI can render."""
    nodes = spec.get("nodes", [])
    preview_nodes = [
        {"id": node.get("id"), "type": node.get("type")}
        for node in nodes
        if isinstance(node, dict)
    ]
    preview_edges: List[Dict[str, Any]] = []
    for idx in range(len(preview_nodes) - 1):
        source = preview_nodes[idx].get("id")
        target = preview_nodes[idx + 1].get("id")
        if source and target:
            preview_edges.append({"source": source, "target": target})
    return {"nodes": preview_nodes, "edges": preview_edges}


async def create_workflow_proposal(
    workflow: Workflow,
    session: Optional[WorkflowChatSession],
    user: User,
    summary: str,
    spec: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> WorkflowProposal:
    """Persist a workflow proposal."""
    normalized_spec = _normalize_spec(spec)
    preview_graph = _preview_from_spec(normalized_spec)

    safe_summary = (summary or "").strip() or "Workflow changes"
    if len(safe_summary) > 512:
        safe_summary = f"{safe_summary[:509]}..."

    proposal = await WorkflowProposal.create(
        workflow=workflow,
        session=session,
        created_by=user,
        summary=safe_summary,
        spec=normalized_spec,
        preview_graph=preview_graph,
        status=WorkflowProposal.STATUS_PENDING,
        metadata=metadata,
    )
    return proposal


async def get_workflow_proposal(
    workflow: Workflow,
    proposal_id: int,
) -> WorkflowProposal:
    """Fetch a workflow proposal."""
    proposal = await WorkflowProposal.get_or_none(id=proposal_id, workflow=workflow)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


async def accept_workflow_proposal(
    workflow: Workflow,
    proposal_id: int,
    *,
    actor: Optional[User] = None,
) -> Tuple[WorkflowProposal, Workflow]:
    """Apply workflow proposal and mark accepted."""
    proposal = await get_workflow_proposal(workflow, proposal_id)
    if proposal.status != WorkflowProposal.STATUS_PENDING:
        raise HTTPException(status_code=400, detail="Proposal is not pending")

    workflow = await proposal.workflow
    normalized_spec = _normalize_spec(proposal.spec or {})
    # draft = await _ensure_workflow_draft(workflow)
    await workflow.fetch_related("draft")
    draft = workflow.draft
    draft.spec = normalized_spec
    draft.revision += 1
    if actor is not None:
        draft.updated_by = actor
    await draft.save()
    workflow.updated_at = datetime.utcnow()
    await workflow.save()

    proposal.status = WorkflowProposal.STATUS_ACCEPTED
    proposal.applied_graph = normalized_spec
    proposal.decided_at = datetime.utcnow()
    await proposal.save()

    return proposal, workflow


async def reject_workflow_proposal(
    workflow: Workflow,
    proposal_id: int,
) -> WorkflowProposal:
    """Reject workflow proposal."""
    proposal = await get_workflow_proposal(workflow, proposal_id)
    if proposal.status != WorkflowProposal.STATUS_PENDING:
        raise HTTPException(status_code=400, detail="Proposal is not pending")

    proposal.status = WorkflowProposal.STATUS_REJECTED
    proposal.decided_at = datetime.utcnow()
    await proposal.save()
    return proposal
