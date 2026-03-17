"""Tests for invalid workflow node model data migration."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_migration_module():
    module_path = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "data_migrations"
        / "migrate_invalid_workflow_node_models.py"
    )
    spec = importlib.util.spec_from_file_location("migrate_invalid_workflow_node_models", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migrate_spec_replaces_invalid_node_models():
    module = _load_migration_module()

    spec = {
        "version": "2",
        "nodes": [
            {
                "id": "agent_1",
                "type": "agent",
                "inputs": {"model": "openai/gpt-4o", "prompt": "Summarize this"},
            },
            {
                "id": "browser_1",
                "type": "browser",
                "task": "Extract the title",
                "model": "anthropic/claude-sonnet-4.5",
            },
            {
                "id": "image_1",
                "type": "image_gen",
                "inputs": {"model": "black-forest-labs/flux.2-flex", "prompt": "A lighthouse at sunset"},
            },
        ],
        "edges": [],
        "triggers": [],
    }

    updated_spec, replacements = module._migrate_spec(spec)

    assert updated_spec["nodes"][0]["inputs"]["model"] == module.AGENT_FALLBACK_MODEL
    assert updated_spec["nodes"][1]["model"] == module.BROWSER_FALLBACK_MODEL
    assert updated_spec["nodes"][2]["inputs"]["model"] == module.IMAGE_GEN_FALLBACK_MODEL
    assert replacements == {"agent": 1, "browser": 1, "image_gen": 1}


def test_migrate_spec_keeps_valid_and_missing_models_unchanged():
    module = _load_migration_module()

    spec = {
        "version": "2",
        "nodes": [
            {
                "id": "agent_1",
                "type": "agent",
                "inputs": {"model": module.AGENT_FALLBACK_MODEL, "prompt": "Summarize this"},
            },
            {
                "id": "browser_1",
                "type": "browser",
                "task": "Extract the title",
                "model": module.BROWSER_FALLBACK_MODEL,
            },
            {
                "id": "browser_2",
                "type": "browser",
                "task": "Extract the title without explicit model",
            },
            {
                "id": "image_1",
                "type": "image_gen",
                "inputs": {"model": module.IMAGE_GEN_FALLBACK_MODEL, "prompt": "A lighthouse at sunset"},
            },
        ],
        "edges": [],
        "triggers": [],
    }

    updated_spec, replacements = module._migrate_spec(spec)

    assert updated_spec["nodes"][0]["inputs"]["model"] == module.AGENT_FALLBACK_MODEL
    assert updated_spec["nodes"][1]["model"] == module.BROWSER_FALLBACK_MODEL
    assert "model" not in updated_spec["nodes"][2]
    assert updated_spec["nodes"][3]["inputs"]["model"] == module.IMAGE_GEN_FALLBACK_MODEL
    assert replacements == {"agent": 0, "browser": 0, "image_gen": 0}


def test_migrate_spec_is_idempotent():
    module = _load_migration_module()

    spec = {
        "version": "2",
        "nodes": [
            {
                "id": "agent_1",
                "type": "agent",
                "inputs": {"model": "invalid/model", "prompt": "Summarize this"},
            },
            {
                "id": "browser_1",
                "type": "browser",
                "task": "Extract the title",
                "model": None,
            },
            {
                "id": "image_1",
                "type": "image_gen",
                "inputs": {"model": ["not", "a", "string"], "prompt": "A lighthouse at sunset"},
            },
        ],
        "edges": [],
        "triggers": [],
    }

    updated_spec, first_replacements = module._migrate_spec(spec)
    rerun_spec, second_replacements = module._migrate_spec(updated_spec)

    assert first_replacements == {"agent": 1, "browser": 1, "image_gen": 1}
    assert second_replacements == {"agent": 0, "browser": 0, "image_gen": 0}
    assert rerun_spec["nodes"][0]["inputs"]["model"] == module.AGENT_FALLBACK_MODEL
    assert rerun_spec["nodes"][1]["model"] == module.BROWSER_FALLBACK_MODEL
    assert rerun_spec["nodes"][2]["inputs"]["model"] == module.IMAGE_GEN_FALLBACK_MODEL
