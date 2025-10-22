"""Shared A2A communication utilities for Seer agents"""

import httpx
import json
import uuid as _uuid
from .config import get_seer_config
from .error_handling import create_error_response, create_success_response


async def resolve_assistant_id(port: int, graph_name: str, timeout: float = 10.0) -> str | None:
    """Resolve a graph name to its assistant UUID on a given server port using search API."""
    base = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            payload = {
                "metadata": {},
                "graph_id": graph_name,
                "limit": 10,
                "offset": 0,
                "sort_by": "assistant_id",
                "sort_order": "asc",
                "select": ["assistant_id"]
            }
            r = await client.post(f"{base}/assistants/search", json=payload)
            if r.status_code >= 400:
                return None
            data = r.json()
            if isinstance(data, list) and data:
                first = data[0]
                aid = first.get("assistant_id") or first.get("id")
                if aid:
                    return aid
    except Exception:
        pass
    return None


async def create_server_thread(port: int, timeout: float | None = None) -> str:
    """Create a LangGraph server thread and return its thread_id."""
    cfg = get_seer_config()
    t = timeout or cfg.a2a_timeout
    base = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(timeout=t) as client:
        resp = await client.post(f"{base}/threads", json={"thread_id": ""})
        if resp.status_code >= 400:
            raise RuntimeError(f"Failed to create server thread: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json() if resp.content else {}
        tid = (data or {}).get("thread_id") or ""
        if not tid:
            raise RuntimeError("Server did not return a thread_id")
        return tid


async def send_a2a_message(target_agent_id: str, target_port: int, message: str, thread_id: str = None) -> str:
    """
    Send a message to another agent using LangGraph A2A protocol.
    
    Args:
        target_agent_id: Assistant ID of target agent
        target_port: Port where target agent is running
        message: Message content to send
        thread_id: Optional thread ID for context
    
    Returns:
        JSON string with response from target agent
    """
    thread_id = thread_id or str(_uuid.uuid4())

    payload = {
        "jsonrpc": "2.0",
        "id": thread_id,
        "method": "message/send",
        "params": {
            "message": {"role": "user", "parts": [{"kind": "text", "text": message}], "messageId": str(_uuid.uuid4())},
            "thread": {"threadId": thread_id}
        }
    }
    
    try:
        config = get_seer_config()
        async with httpx.AsyncClient(timeout=config.a2a_timeout) as client:
            def _extract_a2a_text(data: dict) -> str:
                if not isinstance(data, dict):
                    return ""
                result_field = data.get("result", {}) if isinstance(data.get("result"), dict) else {}
                artifacts = result_field.get("artifacts") if isinstance(result_field.get("artifacts"), list) else []
                if artifacts:
                    parts = artifacts[0].get("parts", []) if isinstance(artifacts[0], dict) else []
                    if parts and isinstance(parts[0], dict):
                        text = parts[0].get("text", "") or parts[0].get("content", "")
                        if text:
                            return text
                # Additional fallbacks
                return (
                    result_field.get("text")
                    or result_field.get("response")
                    or data.get("response")
                    or ""
                )

            def _extract_thread_id(data: dict) -> str:
                if not isinstance(data, dict):
                    return ""
                result_field = data.get("result", {}) if isinstance(data.get("result"), dict) else {}
                tid = (
                    result_field.get("thread_id")
                    or result_field.get("threadId")
                    or (result_field.get("thread") or {}).get("threadId")
                    or (result_field.get("thread") or {}).get("id")
                )
                return tid or ""

            # Resolve to UUID if needed
            def _looks_like_uuid(value: str) -> bool:
                try:
                    _uuid.UUID(value)
                    return True
                except Exception:
                    return False

            # Ensure we are using a valid server thread id
            server_thread_id = thread_id if (isinstance(thread_id, str) and _looks_like_uuid(thread_id)) else None
            if not server_thread_id:
                server_thread_id = await create_server_thread(target_port, timeout=config.a2a_timeout)
            # Bind using A2A spec: thread.threadId
            payload["params"]["thread"]["threadId"] = server_thread_id

            if _looks_like_uuid(target_agent_id):
                target_uuid = target_agent_id
            else:
                target_uuid = await resolve_assistant_id(target_port, target_agent_id, timeout=config.a2a_timeout)
                if not target_uuid:
                    return create_error_response(f"Assistant '{target_agent_id}' not found on port {target_port}")

            # Post to A2A using resolved UUID
            url = f"http://127.0.0.1:{target_port}/a2a/{target_uuid}"
            print(
                "[A2A] sending",
                json.dumps(
                    {
                        "graph_id": target_agent_id,
                        "assistant_id": target_uuid,
                        "port": target_port,
                        "payload": payload,
                    }
                ),
            )
            response = await client.post(url, json=payload, headers={"Accept": "application/json"})
            if response.status_code >= 400:
                return create_error_response(f"A2A error: HTTP {response.status_code} {response.text[:200]}")
            data = response.json()
            text = _extract_a2a_text(data)
            tid = _extract_thread_id(data) or server_thread_id
            if text:
                out = {"response": text}
                if tid:
                    out["thread_id"] = tid
                return create_success_response(out)
            return create_error_response("A2A returned no assistant text")
    except Exception as e:
        return create_error_response(f"Failed to send message via A2A: {str(e)}", e)
