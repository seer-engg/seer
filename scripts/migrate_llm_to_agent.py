#!/usr/bin/env python
"""
Migrate all workflow specs: replace LLM nodes with Agent nodes.

All existing workflow specs that contain nodes with type "llm" are updated
to use type "agent" instead. The agent node supersedes the LLM node and
supports all the same inputs (model, prompt, outputs) plus tool execution.

Usage:
    uv run scripts/migrate_llm_to_agent.py

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


def _migrate_spec(spec: dict) -> tuple[dict, int]:
    """
    Walk spec nodes and replace type "llm" with "agent". Idempotent.

    Returns:
        Tuple of (updated_spec, count_of_changes)
    """
    count = 0
    for node in spec.get("nodes", []):
        if node.get("type") == "llm":
            node["type"] = "agent"
            count += 1
    return spec, count


def _hash_spec(spec: dict) -> str:
    """Compute SHA-256 hex digest of a canonicalised JSON spec."""
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def migrate() -> None:
    await Tortoise.init(config=TORTOISE_ORM)

    total = 0
    versions_updated = 0
    runs_updated = 0
    records_updated = 0
    proposals_updated = 0

    # --- WorkflowVersion: update spec, spec_hash, and clear manifest ---
    async for version in WorkflowVersion.all():
        spec, count = _migrate_spec(dict(version.spec))
        if count:
            version.spec = spec
            version.spec_hash = _hash_spec(spec)
            version.manifest = None
            await version.save(update_fields=["spec", "spec_hash", "manifest"])
            versions_updated += 1
            total += count

    # --- WorkflowRun: update spec snapshot only ---
    async for run in WorkflowRun.all():
        spec, count = _migrate_spec(dict(run.spec))
        if count:
            run.spec = spec
            await run.save(update_fields=["spec"])
            runs_updated += 1
            total += count

    # --- WorkflowRecord: update spec only ---
    async for record in WorkflowRecord.all():
        spec, count = _migrate_spec(dict(record.spec))
        if count:
            record.spec = spec
            await record.save(update_fields=["spec"])
            records_updated += 1
            total += count

    # --- WorkflowProposal: update spec only ---
    async for proposal in WorkflowProposal.all():
        spec, count = _migrate_spec(dict(proposal.spec))
        if count:
            proposal.spec = spec
            await proposal.save(update_fields=["spec"])
            proposals_updated += 1
            total += count

    print("Migration complete.")
    print(f"  LLM nodes migrated to Agent: {total}")
    print(f"  WorkflowVersion rows updated: {versions_updated}")
    print(f"  WorkflowRun rows updated:     {runs_updated}")
    print(f"  WorkflowRecord rows updated:  {records_updated}")
    print(f"  WorkflowProposal rows updated:{proposals_updated}")

    await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(migrate())
    sys.exit(0)
