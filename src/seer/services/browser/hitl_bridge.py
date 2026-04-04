"""
In-process registry for browser HITL asyncio Futures.

Maps (run_id, request_id) to asyncio.Future so the browser agent's
ask_human action can block until a human responds via the resume API.

Thread-safety note: FastAPI and the browser agent share the same asyncio
event loop, so no cross-thread synchronization is needed.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from seer.logger import get_logger

logger = get_logger(__name__)

# Key: (run_id, request_id) → asyncio.Future
_pending_futures: Dict[Tuple[str, str], asyncio.Future] = {}


def register_hitl_wait(run_id: str, request_id: str) -> asyncio.Future:
    """Create and store a Future for a browser HITL wait.

    Args:
        run_id: Workflow run ID.
        request_id: Unique ID for this HITL request (uuid4).

    Returns:
        asyncio.Future that resolves when the human responds.
    """
    key = (run_id, request_id)
    if key in _pending_futures:
        logger.warning("HITL wait already registered for run=%s request=%s, replacing", run_id, request_id)
        _cancel_if_pending(key)

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    _pending_futures[key] = future
    logger.info("Registered HITL wait for run=%s request=%s", run_id, request_id)
    return future


def resolve_hitl_wait(run_id: str, request_id: str, response: Any) -> bool:
    """Resolve a pending HITL Future with the human's response.

    Args:
        run_id: Workflow run ID.
        request_id: Unique ID for this HITL request.
        response: The human's response data.

    Returns:
        True if the Future was found and resolved, False otherwise.
    """
    key = (run_id, request_id)
    future = _pending_futures.pop(key, None)
    if future is None:
        logger.warning("No pending HITL wait for run=%s request=%s", run_id, request_id)
        return False

    if future.done():
        logger.warning("HITL Future already done for run=%s request=%s", run_id, request_id)
        return False

    future.set_result(response)
    logger.info("Resolved HITL wait for run=%s request=%s", run_id, request_id)
    return True


def cancel_hitl_wait(run_id: str, request_id: str) -> bool:
    """Cancel a pending HITL Future.

    Args:
        run_id: Workflow run ID.
        request_id: Unique ID for this HITL request.

    Returns:
        True if the Future was found and cancelled, False otherwise.
    """
    key = (run_id, request_id)
    return _cancel_if_pending(key)


def cancel_all_for_run(run_id: str) -> int:
    """Cancel all pending HITL Futures for a workflow run.

    Used when a workflow run is cancelled.

    Args:
        run_id: Workflow run ID.

    Returns:
        Number of Futures cancelled.
    """
    keys_to_cancel = [k for k in _pending_futures if k[0] == run_id]
    count = 0
    for key in keys_to_cancel:
        if _cancel_if_pending(key):
            count += 1
    return count


def has_pending_wait(run_id: str, request_id: str) -> bool:
    """Check if a HITL wait is pending.

    Args:
        run_id: Workflow run ID.
        request_id: Unique ID for this HITL request.

    Returns:
        True if a Future exists and is not done.
    """
    key = (run_id, request_id)
    future = _pending_futures.get(key)
    return future is not None and not future.done()


def find_request_id_for_run(run_id: str) -> Optional[str]:
    """Find the request_id for a pending HITL wait on a given run.

    Useful when the resume endpoint only knows the run_id and needs
    to look up the active request_id.

    Args:
        run_id: Workflow run ID.

    Returns:
        The request_id if found, None otherwise.
    """
    for (rid, req_id), future in _pending_futures.items():
        if rid == run_id and not future.done():
            return req_id
    return None


def _cancel_if_pending(key: Tuple[str, str]) -> bool:
    """Cancel and remove a Future if it exists and is not done."""
    future = _pending_futures.pop(key, None)
    if future is None:
        return False
    if not future.done():
        future.cancel()
        logger.info("Cancelled HITL wait for run=%s request=%s", key[0], key[1])
        return True
    return False
