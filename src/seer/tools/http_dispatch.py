"""Shared HTTP dispatch logic for API tool base classes."""

from typing import Any, Dict, Optional

import httpx


async def dispatch_http(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: Dict[str, str],
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> httpx.Response:
    """
    Dispatch an HTTP request using the given client.

    Args:
        client: httpx.AsyncClient instance
        method: HTTP method (GET, POST, PATCH, DELETE)
        url: Full request URL
        headers: Request headers
        params: Query parameters
        json_body: JSON request body

    Returns:
        httpx.Response

    Raises:
        ValueError: If method is not supported
    """
    m = method.upper()
    if m == "GET":
        return await client.get(url, headers=headers, params=params)
    if m == "POST":
        return await client.post(url, headers=headers, params=params, json=json_body)
    if m == "PATCH":
        return await client.patch(url, headers=headers, params=params, json=json_body)
    if m == "DELETE":
        return await client.delete(url, headers=headers, params=params)
    raise ValueError(f"Unsupported HTTP method: {method}")
