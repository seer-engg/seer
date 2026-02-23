"""
Database seeding package for OAuth-related data.

Provides tools to export seed data to S3 and import it back into the database.
This enables quick environment setup with pre-configured OAuth connections.

Usage:
    # Export current database to S3
    uv run scripts/seed_export.py

    # Import seed data from S3
    uv run scripts/seed_import.py
"""

from seer.database.seed.exporter import SeedExporter
from seer.database.seed.importer import SeedImporter
from seer.database.seed.s3_client import SeedS3Client
from seer.database.seed.schema import (
    IntegrationResourceRef,
    IntegrationResourceSeed,
    IntegrationSecretSeed,
    OAuthConnectionRef,
    OAuthConnectionSeed,
    SeedData,
    SeedTables,
    UserSeed,
)

__all__ = [
    # Schema models
    "UserSeed",
    "OAuthConnectionSeed",
    "OAuthConnectionRef",
    "IntegrationResourceSeed",
    "IntegrationResourceRef",
    "IntegrationSecretSeed",
    "SeedTables",
    "SeedData",
    # Clients
    "SeedS3Client",
    "SeedExporter",
    "SeedImporter",
]
