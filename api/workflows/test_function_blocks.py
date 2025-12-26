"""
Tests for the function block schema listing endpoint.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.workflows.router import router as workflows_router


app = FastAPI()
app.include_router(workflows_router, prefix="/api")
client = TestClient(app)


def test_list_function_blocks_endpoint_returns_all_core_blocks():
    response = client.get("/api/workflows/blocks/functions")
    assert response.status_code == 200
    data = response.json()
    blocks = data.get("blocks", [])
    assert isinstance(blocks, list)
    block_types = {block.get("type") for block in blocks}

    assert {"llm", "if_else", "for_loop"}.issubset(block_types)


def test_llm_block_schema_contains_defaults_and_config():
    response = client.get("/api/workflows/blocks/functions")
    assert response.status_code == 200
    blocks = response.json()["blocks"]
    llm_block = next(block for block in blocks if block["type"] == "llm")

    defaults = llm_block.get("defaults", {})
    assert defaults.get("user_prompt")
    config_schema = llm_block.get("config_schema", {})
    assert config_schema.get("required") == ["user_prompt"]
    assert "temperature" in config_schema.get("properties", {})

