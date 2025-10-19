"""Shared A2A communication utilities for Seer agents"""

import httpx
import json
import os
from typing import Optional
from .config import get_seer_config
from .error_handling import create_error_response, create_success_response

# Simple in-process assistant validation cache: {(port, assistant_id, version): bool}
_ASSISTANT_OK_CACHE: dict[tuple[int, str, str], bool] = {}


async def _validate_assistant(client: httpx.AsyncClient, port: int, assistant_id: str) -> bool:
    """Preflight assistant on official endpoints; optional version pin."""
    try:
        version = os.getenv("SEER_ASSISTANT_VERSION", "")
        if version:
            latest_url = f"http://127.0.0.1:{port}/assistants/{assistant_id}/latest"
            # best-effort; ignore non-2xx
            try:
                await client.post(f"{latest_url}?version={version}")
            except Exception:
                pass

        g = await client.get(f"http://127.0.0.1:{port}/assistants/{assistant_id}/graph?xray=false")
        if g.status_code >= 400:
            return False
        s = await client.get(f"http://127.0.0.1:{port}/assistants/{assistant_id}/schemas")
        if s.status_code >= 400:
            return False
        return True
    except Exception:
        return False


async def send_a2a_message(target_agent_id: str, target_port: int, message: str, thread_id: str = None) -> str:
    """
    Send a message to another agent using LangGraph A2A protocol.
    
    Args:
        target_agent_id: Assistant ID of target agent (e.g., "orchestrator", "customer_success", "eval_agent")
        target_port: Port where target agent is running
        message: Message content to send
        thread_id: Optional thread ID for context
    
    Returns:
        JSON string with response from target agent
    """
    # Normalize assistant id: allow callers to pass graph name (e.g., "orchestrator")
    # and map to the configured UUID from deployment-config.json when needed.
    # Keep original id for potential fallback
    original_target_id = target_agent_id

    try:
        from .config import get_assistant_id as _get_assistant_id
        import uuid as _uuid
        def _looks_like_uuid(value: str) -> bool:
            try:
                _uuid.UUID(value)
                return True
            except Exception:
                return False
        if not _looks_like_uuid(target_agent_id):
            target_agent_id = _get_assistant_id(target_agent_id)
    except Exception:
        # If mapping fails, proceed as-is; the server may still accept a graph id
        pass

    payload = {
        "jsonrpc": "2.0",
        "id": "",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}]
            },
            "messageId": "",
            "thread": {"threadId": thread_id or ""}
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

            # Build A2A candidate identifiers (try multiple forms)
            a2a_candidates: list[str] = []
            try:
                import uuid as _uuid
                from .config import get_config as _get_config, get_graph_name as _get_graph_name

                def _looks_like_uuid(value: str) -> bool:
                    try:
                        _uuid.UUID(value)
                        return True
                    except Exception:
                        return False

                # Start with original and normalized
                if original_target_id not in a2a_candidates:
                    a2a_candidates.append(original_target_id)
                if target_agent_id != original_target_id and target_agent_id not in a2a_candidates:
                    a2a_candidates.append(target_agent_id)

                cfg = _get_config()
                # If original looks like UUID, reverse map to agent and graph name
                if _looks_like_uuid(original_target_id):
                    for agent_name in cfg.list_agents():
                        if cfg.get_assistant_id(agent_name) == original_target_id:
                            try:
                                graph_name = _get_graph_name(agent_name)
                                if graph_name and graph_name not in a2a_candidates:
                                    a2a_candidates.append(graph_name)
                            except Exception:
                                pass
                            if agent_name not in a2a_candidates:
                                a2a_candidates.append(agent_name)
                            break
                else:
                    # original is likely agent/graph name; also try mapped uuid if available
                    try:
                        mapped_uuid = cfg.get_assistant_id(original_target_id)
                        if mapped_uuid and mapped_uuid not in a2a_candidates:
                            a2a_candidates.append(mapped_uuid)
                    except Exception:
                        pass
            except Exception:
                # Best-effort candidates
                a2a_candidates = [target_agent_id]

            # Pre-validate assistant using official endpoints; choose first that passes
            chosen: Optional[str] = None
            version_pin = os.getenv("SEER_ASSISTANT_VERSION", "")
            for cand_id in a2a_candidates:
                cache_key = (target_port, cand_id, version_pin)
                ok = _ASSISTANT_OK_CACHE.get(cache_key)
                if ok is None:
                    ok = await _validate_assistant(client, target_port, cand_id)
                    _ASSISTANT_OK_CACHE[cache_key] = ok
                if ok:
                    chosen = cand_id
                    break

            if not chosen:
                return create_error_response("Assistant not found or not registered on target port")

            # Try A2A delivery
            try:
                url = f"http://127.0.0.1:{target_port}/a2a/{chosen}"
                response = await client.post(url, json=payload, headers={"Accept": "application/json"})
                if response.status_code < 400:
                    data = response.json()
                    text = _extract_a2a_text(data)
                    if text:
                        return create_success_response({"response": text})
            except Exception:
                pass

            # Stream fallback only (no /invoke, no /graphs)
            if os.getenv("SEER_A2A_FALLBACK", "stream").lower() != "stream":
                return create_success_response({"response": "No response received"})

            s_payload = {
                "assistant_id": chosen,
                "input": {"messages": [{"role": "user", "content": message}]},
                "stream_mode": ["values"],
                "config": {"configurable": {"thread_id": thread_id or ""}},
            }
            s_resp = await client.post(f"http://127.0.0.1:{target_port}/runs/stream", json=s_payload)
            if s_resp.status_code < 400:
                final_response = ""
                for line in s_resp.text.strip().split('\n'):
                    if line.startswith('data: '):
                        try:
                            obj = json.loads(line[6:])
                        except Exception:
                            continue
                        msgs = None
                        if isinstance(obj, dict):
                            vals = obj.get("values", {}) if isinstance(obj.get("values"), dict) else {}
                            if not isinstance(vals, dict):
                                vals = obj.get("value", {}) if isinstance(obj.get("value"), dict) else {}
                            msgs = vals.get("messages") if isinstance(vals, dict) else obj.get("messages")
                        if isinstance(msgs, list):
                            for msg in msgs:
                                if isinstance(msg, dict) and (msg.get("type") == "ai" or msg.get("role") == "assistant"):
                                    text = msg.get("content") or msg.get("text") or ""
                                    if text:
                                        final_response = text
                if final_response:
                    return create_success_response({"response": final_response})

            return create_success_response({"response": "No response received"})
    except Exception as e:
        return create_error_response(f"Failed to send message: {str(e)}", e)
