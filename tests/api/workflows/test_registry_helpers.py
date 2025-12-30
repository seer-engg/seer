import pytest

from api.workflows import services
from tests.api.workflows.shared_data import TEST_SCHEMA_ID


@pytest.mark.asyncio
async def test_registry_helpers(db_user):
    """Confirms registry endpoints expose node types, tools, models, and schemas."""
    node_types = await services.list_node_types()
    assert node_types.node_types

    tools_resp = await services.list_tools(include_schemas=True)
    assert tools_resp.tools and tools_resp.tools[0].input_schema is not None

    models_resp = await services.list_models()
    assert models_resp.models

    schema_resp = await services.resolve_schema(TEST_SCHEMA_ID)
    assert schema_resp.json_schema["type"] == "object"

