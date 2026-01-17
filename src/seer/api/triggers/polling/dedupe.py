from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Optional

JsonDict = Dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Value of type {type(value)!r} is not JSON serializable")


def compute_event_hash(
    *,
    trigger_key: str,
    provider_connection_id: Optional[int],
    envelope: JsonDict,
) -> str:
    """Generate a stable SHA256 digest for dedupe fallback."""

    canonical_payload = {
        "trigger_key": trigger_key,
        "account_id": provider_connection_id,
        "occurred_at": envelope.get("occurred_at"),
        "payload": envelope.get("data"),
    }
    body = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
