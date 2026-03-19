#!/usr/bin/env python
"""
Migrate workflow specs: replace invalid node models with valid defaults.

This script scans all stored workflow specs and normalizes invalid models configured on:
    - AgentNode inputs.model
    - BrowserNode model
    - ImageGenNode inputs.model

Each node type is checked against its canonical allowed-model source in code.
If a node has a configured model that is not allowed for that node type, the
script replaces it with a deterministic valid fallback model for that node.

The migration is idempotent: running it multiple times will only update specs
the first time invalid models are encountered.

Usage:
    uv run scripts/data_migrations/migrate_invalid_workflow_node_models.py [--dry-run]

Tables migrated:
    - WorkflowVersion  (all statuses; spec + spec_hash + manifest cleared)
    - WorkflowRun      (spec snapshot only)
    - WorkflowRecord   (spec only)
    - WorkflowProposal (spec only)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys

from tortoise import Tortoise

from seer.api.workflows.services.catalog import DEFAULT_BROWSER_MODEL_REGISTRY
from seer.core.registry.default_models import DEFAULT_AGENT_MODEL_IDS
from seer.core.schema.models import ALLOWED_IMAGE_GEN_MODELS
from seer.database.config import DATABASE_URL, TORTOISE_ORM
from seer.database.workflow_models import WorkflowProposal, WorkflowRecord, WorkflowRun, WorkflowVersion

DRY_RUN = "--dry-run" in sys.argv

ALLOWED_BROWSER_MODEL_IDS = frozenset(model.id for model in DEFAULT_BROWSER_MODEL_REGISTRY)

AGENT_FALLBACK_MODEL = "z-ai/glm-5"
BROWSER_FALLBACK_MODEL = "qwen/qwen3-vl-8b-thinking"
IMAGE_GEN_FALLBACK_MODEL = "sourceful/riverflow-v2-fast"


def _ensure_valid_fallbacks() -> None:
    if AGENT_FALLBACK_MODEL not in DEFAULT_AGENT_MODEL_IDS:
        raise RuntimeError(f"Agent fallback model '{AGENT_FALLBACK_MODEL}' is not allowed")
    if BROWSER_FALLBACK_MODEL not in ALLOWED_BROWSER_MODEL_IDS:
        raise RuntimeError(f"Browser fallback model '{BROWSER_FALLBACK_MODEL}' is not allowed")
    if IMAGE_GEN_FALLBACK_MODEL not in ALLOWED_IMAGE_GEN_MODELS:
        raise RuntimeError(f"Image generation fallback model '{IMAGE_GEN_FALLBACK_MODEL}' is not allowed")


def _has_allowed_model(value: object, allowed_models: frozenset[str]) -> bool:
    return isinstance(value, str) and value in allowed_models


def _migrate_spec(spec: dict) -> tuple[dict, dict[str, int]]:
    """
    Replace invalid configured models in workflow nodes. Idempotent.

    Returns:
        Tuple of (updated_spec, replacements_by_node_type)
    """
    replacements = {
        "agent": 0,
        "browser": 0,
        "image_gen": 0,
    }

    for node in spec.get("nodes", []):
        if not isinstance(node, dict):
            continue

        node_type = node.get("type")

        if node_type == "agent":
            inputs = node.get("inputs")
            if isinstance(inputs, dict) and "model" in inputs and not _has_allowed_model(inputs.get("model"), DEFAULT_AGENT_MODEL_IDS):
                inputs["model"] = AGENT_FALLBACK_MODEL
                replacements["agent"] += 1

        if node_type == "browser" and "model" in node and not _has_allowed_model(node.get("model"), ALLOWED_BROWSER_MODEL_IDS):
            node["model"] = BROWSER_FALLBACK_MODEL
            replacements["browser"] += 1

        if node_type == "image_gen":
            inputs = node.get("inputs")
            if isinstance(inputs, dict) and "model" in inputs and not _has_allowed_model(inputs.get("model"), ALLOWED_IMAGE_GEN_MODELS):
                inputs["model"] = IMAGE_GEN_FALLBACK_MODEL
                replacements["image_gen"] += 1

    return spec, replacements


def _hash_spec(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def migrate() -> None:
    _ensure_valid_fallbacks()

    mode_label = "[DRY RUN] " if DRY_RUN else ""
    print(f"{mode_label}Starting migration: replace invalid workflow node models")
    print(f"{mode_label}Database URL: {DATABASE_URL}")
    print()

    await Tortoise.init(config=TORTOISE_ORM)

    versions_updated = 0
    runs_updated = 0
    records_updated = 0
    proposals_updated = 0
    total_replacements = {
        "agent": 0,
        "browser": 0,
        "image_gen": 0,
    }

    async for version in WorkflowVersion.all():
        spec, replacements = _migrate_spec(dict(version.spec))
        total_for_version = sum(replacements.values())
        if not total_for_version:
            continue

        versions_updated += 1
        for node_type, count in replacements.items():
            total_replacements[node_type] += count

        print(
            f"  WorkflowVersion id={version.id} status={version.status} "
            f"agent={replacements['agent']} browser={replacements['browser']} image_gen={replacements['image_gen']}"
        )

        if not DRY_RUN:
            version.spec = spec
            version.spec_hash = _hash_spec(spec)
            version.manifest = None
            await version.save(update_fields=["spec", "spec_hash", "manifest"])

    async for run in WorkflowRun.all():
        spec, replacements = _migrate_spec(dict(run.spec))
        total_for_run = sum(replacements.values())
        if not total_for_run:
            continue

        runs_updated += 1
        for node_type, count in replacements.items():
            total_replacements[node_type] += count

        print(
            f"  WorkflowRun id={run.id} "
            f"agent={replacements['agent']} browser={replacements['browser']} image_gen={replacements['image_gen']}"
        )

        if not DRY_RUN:
            run.spec = spec
            await run.save(update_fields=["spec"])

    async for record in WorkflowRecord.all():
        spec, replacements = _migrate_spec(dict(record.spec))
        total_for_record = sum(replacements.values())
        if not total_for_record:
            continue

        records_updated += 1
        for node_type, count in replacements.items():
            total_replacements[node_type] += count

        print(
            f"  WorkflowRecord id={record.id} "
            f"agent={replacements['agent']} browser={replacements['browser']} image_gen={replacements['image_gen']}"
        )

        if not DRY_RUN:
            record.spec = spec
            await record.save(update_fields=["spec"])

    async for proposal in WorkflowProposal.all():
        spec, replacements = _migrate_spec(dict(proposal.spec))
        total_for_proposal = sum(replacements.values())
        if not total_for_proposal:
            continue

        proposals_updated += 1
        for node_type, count in replacements.items():
            total_replacements[node_type] += count

        print(
            f"  WorkflowProposal id={proposal.id} "
            f"agent={replacements['agent']} browser={replacements['browser']} image_gen={replacements['image_gen']}"
        )

        if not DRY_RUN:
            proposal.spec = spec
            await proposal.save(update_fields=["spec"])

    print()
    print(f"{mode_label}Migration complete.")
    print(f"  Agent node models replaced:     {total_replacements['agent']}")
    print(f"  Browser node models replaced:   {total_replacements['browser']}")
    print(f"  Image gen node models replaced: {total_replacements['image_gen']}")
    print(f"  WorkflowVersion rows updated:   {versions_updated}")
    print(f"  WorkflowRun rows updated:       {runs_updated}")
    print(f"  WorkflowRecord rows updated:    {records_updated}")
    print(f"  WorkflowProposal rows updated:  {proposals_updated}")
    if DRY_RUN:
        print()
        print("  (No changes were written — re-run without --dry-run to apply)")

    await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(migrate())
    sys.exit(0)
