from __future__ import annotations

from typing import Dict

_store: Dict[str, str] = {}


def write_note(key: str, value: str) -> str:
    _store[key] = value
    return "ok"


def read_note(key: str) -> str:
    return _store.get(key, "")
