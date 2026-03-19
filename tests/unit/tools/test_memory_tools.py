"""Unit tests for shared workflow memory tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from seer.tools.memory.add import MemoryAddTool
from seer.tools.memory.delete import MemoryDeleteTool
from seer.tools.memory.get import MemoryGetTool
from seer.tools.memory.search import MemorySearchTool
from seer.tools.memory.update import MemoryUpdateTool


pytestmark = pytest.mark.unit


@pytest.fixture
def runtime_context():
    context = MagicMock()
    context.memory_access = MagicMock()
    context.memory_access.add = AsyncMock(return_value={"id": "mem_1"})
    context.memory_access.get = AsyncMock(return_value={"id": "mem_1", "memory": "Stored"})
    context.memory_access.search = AsyncMock(return_value=[{"id": "mem_1"}])
    context.memory_access.update = AsyncMock(return_value={"id": "mem_1", "memory": "Updated"})
    context.memory_access.delete = AsyncMock(return_value=True)
    return context


@pytest.mark.asyncio
async def test_memory_add_tool_calls_runtime_adapter(runtime_context):
    result = await MemoryAddTool().execute(
        None,
        {"memory_bank_id": "mb_1", "content": "Remember this", "infer": False, "metadata": {"source": "manual"}},
        context=runtime_context,
    )

    assert result == {"id": "mem_1"}
    runtime_context.memory_access.add.assert_awaited_once_with(
        "mb_1",
        "Remember this",
        metadata={"source": "manual"},
        infer=False,
    )


@pytest.mark.asyncio
async def test_memory_add_tool_rejects_blank_content(runtime_context):
    with pytest.raises(HTTPException, match="Memory content cannot be empty"):
        await MemoryAddTool().execute(None, {"memory_bank_id": "mb_1", "content": "   "}, context=runtime_context)


@pytest.mark.asyncio
async def test_memory_get_tool_calls_runtime_adapter(runtime_context):
    result = await MemoryGetTool().execute(None, {"memory_bank_id": "mb_1", "memory_id": "mem_1"}, context=runtime_context)
    assert result["id"] == "mem_1"


@pytest.mark.asyncio
async def test_memory_search_tool_calls_runtime_adapter(runtime_context):
    result = await MemorySearchTool().execute(None, {"memory_bank_id": "mb_1", "query": "preferences", "limit": 3}, context=runtime_context)
    assert result == [{"id": "mem_1"}]
    runtime_context.memory_access.search.assert_awaited_once_with("mb_1", query="preferences", limit=3)


@pytest.mark.asyncio
async def test_memory_update_tool_calls_runtime_adapter(runtime_context):
    result = await MemoryUpdateTool().execute(
        None,
        {"memory_bank_id": "mb_1", "memory_id": "mem_1", "content": "Updated", "metadata": {"source": "manual"}},
        context=runtime_context,
    )

    assert result["memory"] == "Updated"
    runtime_context.memory_access.update.assert_awaited_once_with(
        "mb_1",
        "mem_1",
        "Updated",
        metadata={"source": "manual"},
    )


@pytest.mark.asyncio
async def test_memory_delete_tool_calls_runtime_adapter(runtime_context):
    result = await MemoryDeleteTool().execute(None, {"memory_bank_id": "mb_1", "memory_id": "mem_1"}, context=runtime_context)
    assert result == {"deleted": True, "memory_id": "mem_1"}
