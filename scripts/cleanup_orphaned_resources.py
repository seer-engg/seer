#!/usr/bin/env python3
"""
One-time script to clean up existing orphaned resources.
Run after deploying the cascade deletion fix.

This script finds and revokes:
1. IntegrationResource records linked to revoked OAuth connections
2. IntegrationSecret records directly tied to revoked OAuth connections (not via resource)

Usage:
    # Dry run (preview what would be cleaned up)
    python scripts/cleanup_orphaned_resources.py

    # Actually clean up
    python scripts/cleanup_orphaned_resources.py --execute
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from tortoise import Tortoise

# Ensure project modules are importable when running as a standalone script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=wrong-import-position
# Reason: sys.path must be modified before importing project modules
from seer.database import TORTOISE_ORM
from seer.database.models_integrations import IntegrationResource, IntegrationSecret
# pylint: enable=wrong-import-position

logger = logging.getLogger("cleanup_orphaned_resources")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


async def cleanup_orphaned_resources(dry_run: bool = True):
    """Find and revoke resources linked to revoked connections."""
    try:
        await Tortoise.init(config=TORTOISE_ORM)
        logger.info("Database initialized")

        # Find resources with revoked connections
        orphaned = await IntegrationResource.filter(
            status="active",
            oauth_connection__status="revoked"
        ).prefetch_related("user", "oauth_connection")

        logger.info("Found %d orphaned resources", len(orphaned))

        for resource in orphaned:
            logger.info("  - Resource %d: %s/%s", resource.id, resource.provider, resource.resource_type)
            logger.info("    User: %s", resource.user.user_id)
            logger.info("    Connection: %d (revoked)", resource.oauth_connection.id)

            if not dry_run:
                # Revoke the resource
                resource.status = "revoked"
                await resource.save(update_fields=["status", "updated_at"])

                # Revoke all secrets for this resource
                await IntegrationSecret.filter(resource=resource, user=resource.user).update(status="revoked")
                logger.info("    ✓ Revoked resource and its secrets")

        # Find secrets directly on revoked connections (not via resource)
        orphaned_secrets = await IntegrationSecret.filter(
            status="active",
            oauth_connection__status="revoked",
            resource_id__isnull=True
        ).prefetch_related("oauth_connection", "user")

        logger.info("\nFound %d orphaned secrets", len(orphaned_secrets))

        for secret in orphaned_secrets:
            logger.info("  - Secret %d: %s/%s", secret.id, secret.provider, secret.secret_type)
            logger.info("    User: %s", secret.user.user_id)
            logger.info("    Connection: %d (revoked)", secret.oauth_connection.id)
            if not dry_run:
                secret.status = "revoked"
                await secret.save(update_fields=["status", "updated_at"])
                logger.info("    ✓ Revoked")

        if dry_run:
            logger.info("\n[DRY RUN] No changes were made. Run with --execute to apply changes.")
        else:
            logger.info("\n[COMPLETE] All orphaned resources and secrets have been revoked.")

    except Exception as e:
        logger.error("Error during cleanup: %s", e, exc_info=True)
        raise
    finally:
        await Tortoise.close_connections()
        logger.info("Database connections closed")


async def main():
    dry_run = "--execute" not in sys.argv
    if dry_run:
        logger.info("Running in DRY RUN mode. Use --execute to apply changes.")
    else:
        logger.info("Running in EXECUTE mode. Changes will be applied.")

    await cleanup_orphaned_resources(dry_run=dry_run)


if __name__ == "__main__":
    asyncio.run(main())
