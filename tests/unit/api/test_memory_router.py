"""Tests for the memory management router."""

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
    app = FastAPI()
    app.include_router(router, prefix="/api")

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = user
        request.state.db_user = user
        return await call_next(request)

    return app


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


class TestCompatibilityRoutes:
    @patch("seer.api.memory.router.config")
    @patch("seer.api.memory.router.MemoryBankMemoryService")
    @patch("seer.api.memory.router.MemoryBankService")
    async def test_create_memory_success(self, mock_bank_service_cls, mock_bank_memory_service_cls, mock_config):
        mock_config.memory_enabled = True
        bank = MagicMock()
        mock_bank_service = MagicMock()
        mock_bank_service.get_or_create_default_bank = AsyncMock(return_value=bank)
        mock_bank_service_cls.return_value = mock_bank_service

        mock_memory_service = MagicMock()
        mock_memory_service.create_manual_memory = AsyncMock(return_value={
            "id": "mem_1",
            "memory": "User prefers concise answers",
            "user_id": "user-123",
            "created_at": "2026-03-01T00:00:00+00:00",
            "metadata": {"source": "manual"},
        })
        mock_bank_memory_service_cls.return_value = mock_memory_service

        app = _make_test_app(_make_user())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/memory", json={"memory": "  User prefers concise answers  "})

        assert response.status_code == 200
        assert response.json()["memory"]["memory"] == "User prefers concise answers"
        mock_memory_service.create_manual_memory.assert_awaited_once()
        assert mock_memory_service.create_manual_memory.await_args.args[2] == "User prefers concise answers"

    @patch("seer.api.memory.router.config")
    async def test_create_memory_rejects_blank_content(self, mock_config):
        mock_config.memory_enabled = True

        app = _make_test_app(_make_user())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/memory", json={"memory": "   "})

        assert response.status_code == 422
        assert response.json()["detail"] == "Memory content cannot be empty"

    @patch("seer.api.memory.router.config")
    @patch("seer.api.memory.router.MemoryBankMemoryService")
    @patch("seer.api.memory.router.MemoryBankService")
    async def test_update_memory_success(self, mock_bank_service_cls, mock_bank_memory_service_cls, mock_config):
        mock_config.memory_enabled = True
        bank = MagicMock()
        mock_bank_service = MagicMock()
        mock_bank_service.get_or_create_default_bank = AsyncMock(return_value=bank)
        mock_bank_service_cls.return_value = mock_bank_service

        mock_memory_service = MagicMock()
        mock_memory_service.get_memory = AsyncMock(return_value={
            "id": "mem_1",
            "memory": "Original memory",
            "user_id": "user-123",
            "metadata": {"source": "chat_session"},
        })
        mock_memory_service.update_memory = AsyncMock(return_value={
            "id": "mem_1",
            "memory": "Updated memory",
            "user_id": "user-123",
            "metadata": {"source": "chat_session"},
        })
        mock_bank_memory_service_cls.return_value = mock_memory_service

        app = _make_test_app(_make_user())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put("/api/memory/mem_1", json={"memory": " Updated memory "})

        assert response.status_code == 200
        assert response.json()["memory"]["memory"] == "Updated memory"
        await_args = mock_memory_service.update_memory.await_args
        args, kwargs = await_args
        assert args[2] == "mem_1"
        assert kwargs["content"] == "Updated memory"
        assert kwargs["metadata"] == {"source": "chat_session"}

    @patch("seer.api.memory.router.config")
    @patch("seer.api.memory.router.MemoryBankMemoryService")
    @patch("seer.api.memory.router.MemoryBankService")
    async def test_update_memory_returns_404_for_foreign_memory(self, mock_bank_service_cls, mock_bank_memory_service_cls, mock_config):
        mock_config.memory_enabled = True
        bank = MagicMock()
        mock_bank_service = MagicMock()
        mock_bank_service.get_or_create_default_bank = AsyncMock(return_value=bank)
        mock_bank_service_cls.return_value = mock_bank_service

        mock_memory_service = MagicMock()
        mock_memory_service.get_memory = AsyncMock(return_value=None)
        mock_bank_memory_service_cls.return_value = mock_memory_service

        app = _make_test_app(_make_user())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put("/api/memory/mem_1", json={"memory": "Updated memory"})

        assert response.status_code == 404
        assert "does not belong to you" in response.json()["detail"]


class TestBankRoutes:
    @patch("seer.api.memory.router.config")
    @patch("seer.api.memory.router.MemoryBankMemoryService")
    @patch("seer.api.memory.router.MemoryBankService")
    async def test_list_memory_banks(self, mock_bank_service_cls, mock_bank_memory_service_cls, mock_config):
        mock_config.memory_enabled = True
        user = _make_user()
        org = MagicMock()
        org.id = 99
        bank = MagicMock()
        bank.public_id = "mb_1"
        bank.name = "Product"
        bank.description = "Workspace memory"
        bank.is_default = True
        bank.created_at.isoformat.return_value = "2026-03-01T00:00:00+00:00"
        bank.updated_at.isoformat.return_value = "2026-03-02T00:00:00+00:00"
        bank.fetch_related = AsyncMock()

        mock_bank_service = MagicMock()
        mock_bank_service.list_banks = AsyncMock(return_value=[bank])
        mock_bank_service_cls.return_value = mock_bank_service

        mock_bank_memory_service = MagicMock()
        mock_bank_memory_service.get_all = AsyncMock(return_value=[{"id": "mem_1"}, {"id": "mem_2"}])
        mock_bank_memory_service_cls.return_value = mock_bank_memory_service

        app = _make_test_app(user)

        @app.middleware("http")
        async def add_org(request, call_next):
            request.state.organization = org
            request.state.membership = MagicMock()
            return await call_next(request)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/memory/banks")

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["memory_bank_id"] == "mb_1"
        assert payload["items"][0]["memory_count"] == 2
