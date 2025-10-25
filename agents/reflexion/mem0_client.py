import os
from typing import Dict, Any, List, Optional

import requests


MEM0_BASE_URL = "https://api.mem0.ai"


def _mem0_headers() -> Dict[str, str]:
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        raise RuntimeError("MEM0_API_KEY not set in environment")
    return {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }


def mem0_add_memory(messages: List[Dict[str, str]], user_id: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "messages": messages,
    }
    if user_id:
        payload["user_id"] = user_id
    resp = requests.post(
        f"{MEM0_BASE_URL}/v1/memories/",
        headers=_mem0_headers(),
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def mem0_search_memories(query: str, user_id: str, timeout: int = 30) -> List[Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "query": query,
        "filters": {"user_id": user_id},
    }

    resp = requests.post(
        f"{MEM0_BASE_URL}/v2/memories/search/",
        headers=_mem0_headers(),
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    # Normalize results into a list of dicts
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
        if "memories" in data and isinstance(data["memories"], list):
            return data["memories"]
        return [data]
    if isinstance(data, list):
        return data
    return [ {"content": str(data)} ]


