"""Unit tests for the default-bank compatibility wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.services.memory.user_memory import UserMemoryService, get_user_memory_service


pytestmark = pytest.mark.unit


@pytest.fixture
def resolved_user():
    user = MagicMock()
    user.user_id = "user_123"
    return user


@pytest.fixture
def resolved_bank():
    bank = MagicMock()
    bank.public_id = "mb_1"
    return bank


@pytest.fixture
def memory_service(resolved_user, resolved_bank):
    service = UserMemoryService()
    service._client = MagicMock()
    service._resolve_default_bank = AsyncMock(return_value=(resolved_user, resolved_bank))
    service._bank_memory_service = MagicMock()
    service._bank_memory_service.is_available = True
    return service


class TestUserMemoryService:
    async def test_is_available_delegates_to_bank_service(self, memory_service):
        assert memory_service.is_available is True
        memory_service._bank_memory_service.is_available = False
        assert memory_service.is_available is False

    async def test_add_memory_delegates_to_default_bank(self, memory_service, resolved_user, resolved_bank):
        memory_service._bank_memory_service.add_memory = AsyncMock(return_value={"results": [{"id": "mem_1"}]})

        result = await memory_service.add_memory(
            user_id="user_123",
            content="Remember this",
            metadata={"session_id": 42},
            infer=False,
        )

        assert result == {"results": [{"id": "mem_1"}]}
        memory_service._resolve_default_bank.assert_awaited_once_with("user_123")
        memory_service._bank_memory_service.add_memory.assert_awaited_once_with(
            resolved_user,
            resolved_bank,
            content="Remember this",
            metadata={"session_id": 42},
            infer=False,
        )

    async def test_search_delegates_to_default_bank(self, memory_service, resolved_user, resolved_bank):
        memories = [{"id": "mem_1", "memory": "User prefers Slack"}]
        memory_service._bank_memory_service.search = AsyncMock(return_value=memories)

        result = await memory_service.search("user_123", "slack", limit=3, filters={"workflow_id": "wf_1"})

        assert result == memories
        memory_service._bank_memory_service.search.assert_awaited_once_with(
            resolved_user,
            resolved_bank,
            query="slack",
            limit=3,
            filters={"workflow_id": "wf_1"},
        )

    async def test_get_all_delegates_to_default_bank(self, memory_service, resolved_user, resolved_bank):
        memories = [{"id": "mem_1"}]
        memory_service._bank_memory_service.get_all = AsyncMock(return_value=memories)

        result = await memory_service.get_all("user_123", filters={"source": "manual"})

        assert result == memories
        memory_service._bank_memory_service.get_all.assert_awaited_once_with(
            resolved_user,
            resolved_bank,
            filters={"source": "manual"},
        )

    async def test_get_memory_with_user_id_is_bank_scoped(self, memory_service, resolved_user, resolved_bank):
        memory = {"id": "mem_1", "user_id": "user_123"}
        memory_service._bank_memory_service.get_memory = AsyncMock(return_value=memory)

        result = await memory_service.get_memory("mem_1", user_id="user_123")

        assert result == memory
        memory_service._bank_memory_service.get_memory.assert_awaited_once_with(resolved_user, resolved_bank, "mem_1")

    async def test_get_memory_without_user_id_uses_raw_client(self, memory_service):
        memory_service._client.get.return_value = {"id": "mem_1"}

        result = await memory_service.get_memory("mem_1")

        assert result == {"id": "mem_1"}
        memory_service._resolve_default_bank.assert_not_awaited()

    async def test_update_memory_with_user_id_delegates(self, memory_service, resolved_user, resolved_bank):
        updated = {"id": "mem_1", "memory": "Updated"}
        memory_service._bank_memory_service.update_memory = AsyncMock(return_value=updated)

        result = await memory_service.update_memory("mem_1", "Updated", metadata={"source": "manual"}, user_id="user_123")

        assert result == updated
        memory_service._bank_memory_service.update_memory.assert_awaited_once_with(
            resolved_user,
            resolved_bank,
            "mem_1",
            content="Updated",
            metadata={"source": "manual"},
        )

    async def test_delete_memory_with_user_id_delegates(self, memory_service, resolved_user, resolved_bank):
        memory_service._bank_memory_service.delete_memory = AsyncMock(return_value=True)

        result = await memory_service.delete_memory("mem_1", user_id="user_123")

        assert result is True
        memory_service._bank_memory_service.delete_memory.assert_awaited_once_with(resolved_user, resolved_bank, "mem_1")

    async def test_get_context_for_prompt_delegates(self, memory_service, resolved_user, resolved_bank):
        memory_service._bank_memory_service.get_context_for_prompt = AsyncMock(return_value="## User Context")

        result = await memory_service.get_context_for_prompt("user_123", current_query="notifications", max_memories=7)

        assert result == "## User Context"
        memory_service._bank_memory_service.get_context_for_prompt.assert_awaited_once_with(
            resolved_user,
            resolved_bank,
            current_query="notifications",
            max_memories=7,
        )


class TestSingleton:
    def test_get_user_memory_service_returns_same_instance(self):
        import seer.services.memory.user_memory as module

        module._SERVICE_INSTANCE = None
        service_one = get_user_memory_service()
        service_two = get_user_memory_service()

        assert service_one is service_two


class TestResolveDefaultBank:
    @pytest.mark.asyncio
    async def test_resolve_default_bank_looks_up_user_and_bank(self):
        service = UserMemoryService()
        fake_user = MagicMock()
        fake_user.user_id = "user_123"
        fake_bank = MagicMock()

        with patch("seer.services.memory.user_memory.User.get_or_none", AsyncMock(return_value=fake_user)), \
             patch.object(service._bank_service, "get_or_create_default_bank", AsyncMock(return_value=fake_bank)):
            user, bank = await service._resolve_default_bank("user_123")

        assert user is fake_user
        assert bank is fake_bank
