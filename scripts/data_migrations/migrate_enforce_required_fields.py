#!/usr/bin/env python
"""
Migrate all workflow specs: ensure every JSON output schema lists all its
properties in the `required` array.

When an AgentNode has outputs.mode=json (or any node has expect_outputs.mode=json),
the LLM uses `required` to decide which fields are mandatory. Properties absent
from `required` may be silently omitted, causing downstream ${node.field}
references to crash with EvaluationError at runtime.

This script walks every stored spec and adds any missing property names to
`required`, recursing into nested object schemas and array item schemas.
It is idempotent: running it twice leaves specs unchanged on the second pass.

Usage:
    uv run scripts/migrate_enforce_required_fields.py [--dry-run]

Tables migrated:
    - WorkflowVersion  (spec + spec_hash + manifest cleared)
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

from seer.database.config import TORTOISE_ORM
from seer.database.workflow_models import WorkflowProposal, WorkflowRecord, WorkflowRun, WorkflowVersion

DRY_RUN = "--dry-run" in sys.argv


# ---------------------------------------------------------------------------
# Core fixer — mirrors _enforce_all_properties_required but mutates instead
# of raising, and returns a count of schemas that were patched.
# ---------------------------------------------------------------------------


def _fix_schema_required(schema: object) -> int:
    """
    Recursively ensure every JSON Schema object lists all its `properties` in
    `required`. Mutates `schema` in place.

    Returns the number of individual schema objects that were patched.
    """
    if not isinstance(schema, dict):
        return 0

    patched = 0

    if schema.get("type") == "object" and "properties" in schema:
        declared = set(schema["properties"].keys())
        existing_required = set(schema.get("required") or [])
        missing = declared - existing_required
        if missing:
            schema["required"] = sorted(existing_required | missing)
            patched += 1
        # Recurse into property sub-schemas
        for prop_schema in schema["properties"].values():
            patched += _fix_schema_required(prop_schema)

    # Recurse into array items
    if "items" in schema and isinstance(schema["items"], dict):
        patched += _fix_schema_required(schema["items"])

    return patched


def _fix_output_contract(contract: object) -> int:
    """
    Given a raw output-contract dict (as stored in JSON), fix its inline schema
    if present. SchemaRef objects (those with an `"id"` key) are left untouched
    because they're managed by the schema registry, not inline.

    The DB stores InlineSchema in two possible formats depending on how the spec
    was saved:
      - Via model_dump(by_alias=True) / raw frontend JSON:
          {"schema": {"schema": {...actual json schema...}}}
      - Via model_dump() without by_alias (most API paths use _spec_to_dict):
          {"schema": {"json_schema": {...actual json schema...}}}

    Both formats are handled here.

    Returns the number of schema objects patched.
    """
    if not isinstance(contract, dict):
        return 0
    if contract.get("mode") != "json":
        return 0

    schema_field = contract.get("schema")
    if not isinstance(schema_field, dict):
        return 0

    # SchemaRef: {"id": "..."}
    if "id" in schema_field:
        return 0

    # InlineSchema: inner schema is under "json_schema" (model_dump field name)
    # or "schema" (alias / raw frontend format)
    inner = schema_field.get("json_schema") or schema_field.get("schema")
    if not isinstance(inner, dict):
        return 0

    return _fix_schema_required(inner)


# Node fields that may carry an OutputContract
_OUTPUT_FIELDS = ("outputs", "expect_outputs")


def _migrate_spec(spec: dict) -> tuple[dict, int]:
    """
    Walk spec nodes and fix all inline JSON output schemas.

    Returns:
        (updated_spec, total_schemas_patched)
    """
    total = 0
    for node in spec.get("nodes", []):
        for field in _OUTPUT_FIELDS:
            contract = node.get(field)
            if contract:
                total += _fix_output_contract(contract)
    return spec, total


def _hash_spec(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------


async def migrate() -> None:  # pylint: disable=too-complex # migration scripts are intentionally verbose
    await Tortoise.init(config=TORTOISE_ORM)

    mode_label = "[DRY RUN] " if DRY_RUN else ""
    print(f"{mode_label}Starting migration: enforce all properties in required for JSON output schemas")
    print()

    versions_updated = runs_updated = records_updated = proposals_updated = 0
    total_schemas_patched = 0

    # --- WorkflowVersion ---
    async for version in WorkflowVersion.all():
        spec, count = _migrate_spec(dict(version.spec))
        if count:
            total_schemas_patched += count
            versions_updated += 1
            print(f"  WorkflowVersion id={version.id} — patched {count} schema(s)")
            if not DRY_RUN:
                version.spec = spec
                version.spec_hash = _hash_spec(spec)
                version.manifest = None
                await version.save(update_fields=["spec", "spec_hash", "manifest"])

    # --- WorkflowRun ---
    async for run in WorkflowRun.all():
        spec, count = _migrate_spec(dict(run.spec))
        if count:
            total_schemas_patched += count
            runs_updated += 1
            print(f"  WorkflowRun id={run.id} — patched {count} schema(s)")
            if not DRY_RUN:
                run.spec = spec
                await run.save(update_fields=["spec"])

    # --- WorkflowRecord ---
    async for record in WorkflowRecord.all():
        spec, count = _migrate_spec(dict(record.spec))
        if count:
            total_schemas_patched += count
            records_updated += 1
            print(f"  WorkflowRecord id={record.id} — patched {count} schema(s)")
            if not DRY_RUN:
                record.spec = spec
                await record.save(update_fields=["spec"])

    # --- WorkflowProposal ---
    async for proposal in WorkflowProposal.all():
        spec, count = _migrate_spec(dict(proposal.spec))
        if count:
            total_schemas_patched += count
            proposals_updated += 1
            print(f"  WorkflowProposal id={proposal.id} — patched {count} schema(s)")
            if not DRY_RUN:
                proposal.spec = spec
                await proposal.save(update_fields=["spec"])

    print()
    print(f"{mode_label}Migration complete.")
    print(f"  Total schema objects patched:  {total_schemas_patched}")
    print(f"  WorkflowVersion rows updated:  {versions_updated}")
    print(f"  WorkflowRun rows updated:      {runs_updated}")
    print(f"  WorkflowRecord rows updated:   {records_updated}")
    print(f"  WorkflowProposal rows updated: {proposals_updated}")
    if DRY_RUN:
        print()
        print("  (No changes were written — re-run without --dry-run to apply)")

    await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(migrate())
    sys.exit(0)
