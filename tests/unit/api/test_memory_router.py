"""Tests for memory management router."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from seer.api.memory.router import router


def _make_user(user_id: str = "user-123"):
    user = MagicMock()
    user.user_id = user_id
    return user


def _make_test_app(user) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api")

    @test_app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = user
        return await call_next(request)

    return test_app


@pytest.mark.asyncio
@pytest.mark.unit
class TestCreateMemory:
    """Test POST /api/memory endpoint."""

    @patch("seer.api.memory.router.config")
    @patch("seer.api.memory.router.UserMemoryService")
    async def test_create_memory_success(self, mock_service_cls, mock_config):
        mock_config.memory_enabled = True
        mock_service = MagicMock()
        mock_service.create_manual_memory = AsyncMock(return_value={
            "id": "mem_1",
            "memory": "User prefers concise answers",
            "user_id": "user-123",
            "created_at": "2026-03-01T00:00:00+00:00",
            "metadata": {"source": "manual"},
        })
        mock_service_cls.return_value = mock_service

        app = _make_test_app(_make_user())
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/memory", json={"memory": "  User prefers concise answers  "})

        assert response.status_code == 200
        payload = response.json()
        assert payload["memory"]["memory"] == "User prefers concise answers"
        mock_service.create_manual_memory.assert_awaited_once_with(
            user_id="user-123",
            content="User prefers concise answers",
        )

    @patch("seer.api.memory.router.config")
    async def test_create_memory_rejects_blank_content(self, mock_config):
        mock_config.memory_enabled = True

        app = _make_test_app(_make_user())
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/memory", json={"memory": "   "})

        assert response.status_code == 422
        assert response.json()["detail"] == "Memory content cannot be empty"


@pytest.mark.asyncio
@pytest.mark.unit
class TestUpdateMemory:
    """Test PUT /api/memory/{memory_id} endpoint."""

    @patch("seer.api.memory.router.config")
    @patch("seer.api.memory.router.UserMemoryService")
    async def test_update_memory_success(self, mock_service_cls, mock_config):
        mock_config.memory_enabled = True
        existing_memory = {
            "id": "mem_1",
            "memory": "Original memory",
            "user_id": "user-123",
            "created_at": "2026-03-01T00:00:00+00:00",
            "metadata": {"source": "chat_session", "session_id": 99},
        }
        updated_memory = {
            "id": "mem_1",
            "memory": "Updated memory",
            "user_id": "user-123",
            "created_at": "2026-03-01T00:00:00+00:00",
            "updated_at": "2026-03-02T00:00:00+00:00",
            "metadata": {"source": "chat_session", "session_id": 99, "edited_at": "2026-03-02T00:00:00+00:00"},
        }
        mock_service = MagicMock()
        mock_service.get_memory = AsyncMock(return_value=existing_memory)
        mock_service.update_memory = AsyncMock(return_value=updated_memory)
        mock_service_cls.return_value = mock_service

        app = _make_test_app(_make_user())
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put("/api/memory/mem_1", json={"memory": " Updated memory "})

        assert response.status_code == 200
        payload = response.json()
        assert payload["memory"]["memory"] == "Updated memory"
        mock_service.update_memory.assert_awaited_once_with(
            memory_id="mem_1",
            content="Updated memory",
            metadata={"source": "chat_session"},
        )

    @patch("seer.api.memory.router.config")
    @patch("seer.api.memory.router.UserMemoryService")
    async def test_update_memory_returns_404_for_foreign_memory(self, mock_service_cls, mock_config):
        mock_config.memory_enabled = True
        mock_service = MagicMock()
        mock_service.get_memory = AsyncMock(return_value={
            "id": "mem_1",
            "memory": "Someone else's memory",
            "user_id": "other-user",
            "created_at": "2026-03-01T00:00:00+00:00",
        })
        mock_service_cls.return_value = mock_service

        app = _make_test_app(_make_user())
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put("/api/memory/mem_1", json={"memory": "Updated memory"})

        assert response.status_code == 404
        assert "does not belong to you" in response.json()["detail"]
