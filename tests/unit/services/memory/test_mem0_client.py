"""Unit tests for Mem0 client initialization helpers."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from seer.services.memory import mem0_client


pytestmark = pytest.mark.unit


def teardown_function() -> None:
    mem0_client.reset_mem0_client()


def test_get_mem0_client_disables_telemetry_store_after_init() -> None:
    fake_client = MagicMock()
    fake_client._telemetry_vector_store = object()

    fake_memory_class = MagicMock()
    fake_memory_class.from_config.return_value = fake_client
    fake_module = types.SimpleNamespace(Memory=fake_memory_class)

    with patch.object(mem0_client.config, "memory_enabled", True), \
         patch.object(mem0_client, "_build_mem0_config", return_value={"llm": {"provider": "test"}, "embedder": {"provider": "test"}, "vector_store": {"provider": "pgvector"}}), \
         patch.dict("sys.modules", {"mem0": fake_module}):
        client = mem0_client.get_mem0_client()

    assert client is fake_client
    assert fake_client._telemetry_vector_store is None
