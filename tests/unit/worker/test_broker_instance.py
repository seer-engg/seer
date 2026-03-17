"""Unit tests for worker.broker_instance module."""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from seer.config import config


def _reload_broker_instance():
    module_name = "seer.worker.broker_instance"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.unit
def test_broker_passes_configured_idle_timeout():
    """RedisStreamBroker should use the configured reclaim timeout."""
    fake_backend = MagicMock()
    fake_broker = MagicMock()
    fake_broker.with_result_backend.return_value = fake_broker

    with patch.object(config, "redis_url", "redis://localhost:6379/0"), \
         patch.object(config, "redis_max_connections", 25), \
         patch.object(config, "redis_socket_timeout", 5), \
         patch.object(config, "redis_socket_connect_timeout", 5), \
         patch.object(config, "redis_socket_keepalive", False), \
         patch.object(config, "redis_health_check_interval", 30), \
         patch.object(config, "nexus_chat_timeout_seconds", 900), \
         patch.object(config, "redis_stream_idle_timeout_ms", 3600000), \
         patch("taskiq_redis.RedisAsyncResultBackend", return_value=fake_backend), \
         patch("taskiq_redis.RedisStreamBroker", return_value=fake_broker) as mock_broker_cls:

        _reload_broker_instance()

    assert mock_broker_cls.call_args.kwargs["idle_timeout"] == 3600000
    assert fake_broker.with_result_backend.call_args.args == (fake_backend,)


@pytest.mark.unit
def test_warns_when_idle_timeout_is_not_above_chat_timeout():
    """Startup should warn when broker reclaim timeout can preempt Nexus timeout."""
    fake_backend = MagicMock()
    fake_broker = MagicMock()
    fake_broker.with_result_backend.return_value = fake_broker
    fake_logger = MagicMock()

    with patch.object(config, "redis_url", "redis://localhost:6379/0"), \
         patch.object(config, "redis_max_connections", 25), \
         patch.object(config, "redis_socket_timeout", 5), \
         patch.object(config, "redis_socket_connect_timeout", 5), \
         patch.object(config, "redis_socket_keepalive", False), \
         patch.object(config, "redis_health_check_interval", 30), \
         patch.object(config, "nexus_chat_timeout_seconds", 900), \
         patch.object(config, "redis_stream_idle_timeout_ms", 900000), \
         patch("seer.logger.get_logger", return_value=fake_logger), \
         patch("taskiq_redis.RedisAsyncResultBackend", return_value=fake_backend), \
         patch("taskiq_redis.RedisStreamBroker", return_value=fake_broker):

        _reload_broker_instance()

    assert fake_logger.warning.called
    warning_message = fake_logger.warning.call_args.args[0]
    assert "idle timeout" in warning_message.lower()
