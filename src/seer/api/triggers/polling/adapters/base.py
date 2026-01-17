from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from seer.database import User, OAuthConnection ,TriggerSubscription

JsonDict = Dict[str, Any]


@dataclass(slots=True)
class PollContext:
    """Context shared with adapters when polling a subscription."""

    subscription: TriggerSubscription
    user: User
    connection: Optional[OAuthConnection] = None
    access_token: Optional[str] = None


@dataclass(slots=True)
class PolledEvent:
    """Normalized event emitted by adapters."""

    payload: JsonDict
    raw: Optional[JsonDict] = None
    provider_event_id: Optional[str] = None
    occurred_at: Optional[datetime] = None


@dataclass(slots=True)
class PollResult:
    """Adapter response with new cursor + normalized events."""

    events: List[PolledEvent]
    cursor: JsonDict
    has_more: bool = False
    rate_limit_hint: Optional[int] = None


class PollAdapterError(Exception):
    """Adapter-level exception that carries retry/backoff hints."""

    def __init__(
        self,
        message: str,
        *,
        backoff_seconds: Optional[int] = None,
        permanent: bool = False,
        detail: Optional[JsonDict] = None,
    ) -> None:
        super().__init__(message)
        self.backoff_seconds = backoff_seconds
        self.permanent = permanent
        self.detail = detail or {}


class PollAdapter(Protocol):
    """Interface implemented by provider-specific polling adapters."""

    trigger_key: str

    async def bootstrap_cursor(self, ctx: PollContext) -> JsonDict:
        """Initialize provider cursor for first poll."""

    async def poll(self, ctx: PollContext, cursor: JsonDict) -> PollResult:
        """Return normalized events + new cursor."""


class PollAdapterRegistry:
    """Simple in-memory registry keyed by trigger_key."""

    def __init__(self) -> None:
        self._adapters: Dict[str, PollAdapter] = {}

    def register(self, adapter: PollAdapter) -> None:
        existing = self._adapters.get(adapter.trigger_key)
        if existing is not None and existing is not adapter:
            raise ValueError(f"Poll adapter already registered for {adapter.trigger_key}")
        self._adapters[adapter.trigger_key] = adapter

    def get(self, trigger_key: str) -> Optional[PollAdapter]:
        return self._adapters.get(trigger_key)


adapter_registry = PollAdapterRegistry()


def register_adapter(adapter: PollAdapter) -> PollAdapter:
    adapter_registry.register(adapter)
    return adapter
