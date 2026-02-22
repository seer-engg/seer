"""
Seed data exporter.

Exports User, OAuthConnection, IntegrationResource, and IntegrationSecret
data from the database to S3 as JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from seer.config import config
from seer.database import User
from seer.database.models_integrations import IntegrationResource, IntegrationSecret
from seer.database.models_oauth import OAuthConnection
from seer.database.seed.s3_client import SeedS3Client
from seer.database.seed.schema import (
    IntegrationResourceSeed,
    IntegrationSecretSeed,
    IntegrationResourceRef,
    OAuthConnectionRef,
    OAuthConnectionSeed,
    SeedData,
    SeedTables,
    UserSeed,
)
from seer.logger import get_logger

logger = get_logger("seer.database.seed.exporter")


def _merge_by_key(existing: list[dict], new: list[dict], key_fields: list[str]) -> list[dict]:
    """
    Merge two lists of dicts, using key_fields to identify duplicates.
    New records override existing ones with matching keys.
    """
    # Build index of existing records by key
    def make_key(record: dict) -> tuple:
        return tuple(record.get(f) for f in key_fields)

    merged = {make_key(r): r for r in existing}

    # Upsert new records
    for record in new:
        merged[make_key(record)] = record

    return list(merged.values())


class SeedExporter:
    """Export database seed data to S3."""

    def __init__(self, s3_client: Optional[SeedS3Client] = None):
        """
        Initialize the exporter.

        Args:
            s3_client: S3 client for uploads. Created automatically if not provided.
        """
        self.s3_client = s3_client or SeedS3Client()

    async def export_users(self) -> list[UserSeed]:
        """Export all users."""
        users = await User.all()
        logger.debug("Found %d users to export", len(users))

        return [
            UserSeed(
                user_id=u.user_id,
                email=u.email,
                first_name=u.first_name,
                last_name=u.last_name,
                claims=u.claims,
                signup_source=u.signup_source,
                default_workflow_creation_mode=u.default_workflow_creation_mode,
            )
            for u in users
        ]

    async def export_oauth_connections(self) -> list[OAuthConnectionSeed]:
        """Export all OAuth connections with user references."""
        connections = await OAuthConnection.all().prefetch_related("user")
        logger.debug("Found %d OAuth connections to export", len(connections))

        return [
            OAuthConnectionSeed(
                user_id=c.user.user_id,
                provider=c.provider,
                provider_account_id=c.provider_account_id,
                access_token_enc=c.access_token_enc,
                refresh_token_enc=c.refresh_token_enc,
                expires_at=c.expires_at,
                scopes=c.scopes,
                token_type=c.token_type,
                provider_metadata=c.provider_metadata,
                status=c.status,
            )
            for c in connections
        ]

    async def export_integration_resources(self) -> list[IntegrationResourceSeed]:
        """Export all integration resources with FK references."""
        resources = await IntegrationResource.all().prefetch_related("user", "oauth_connection")
        logger.debug("Found %d integration resources to export", len(resources))

        result = []
        for r in resources:
            oauth_ref = None
            if r.oauth_connection:
                oauth_ref = OAuthConnectionRef(
                    provider=r.oauth_connection.provider,
                    provider_account_id=r.oauth_connection.provider_account_id,
                )

            result.append(
                IntegrationResourceSeed(
                    user_id=r.user.user_id,
                    oauth_connection_ref=oauth_ref,
                    provider=r.provider,
                    resource_type=r.resource_type,
                    resource_id=r.resource_id,
                    resource_key=r.resource_key,
                    name=r.name,
                    resource_metadata=r.resource_metadata,
                    status=r.status,
                )
            )

        return result

    async def export_integration_secrets(self) -> list[IntegrationSecretSeed]:
        """Export all integration secrets with FK references."""
        secrets = await IntegrationSecret.all().prefetch_related(
            "user", "oauth_connection", "resource"
        )
        logger.debug("Found %d integration secrets to export", len(secrets))

        result = []
        for s in secrets:
            oauth_ref = None
            if s.oauth_connection:
                oauth_ref = OAuthConnectionRef(
                    provider=s.oauth_connection.provider,
                    provider_account_id=s.oauth_connection.provider_account_id,
                )

            resource_ref = None
            if s.resource:
                resource_ref = IntegrationResourceRef(
                    provider=s.resource.provider,
                    resource_type=s.resource.resource_type,
                    resource_id=s.resource.resource_id,
                )

            result.append(
                IntegrationSecretSeed(
                    user_id=s.user.user_id,
                    provider=s.provider,
                    oauth_connection_ref=oauth_ref,
                    resource_ref=resource_ref,
                    secret_type=s.secret_type,
                    name=s.name,
                    value_enc=s.value_enc,
                    value_fingerprint=s.value_fingerprint,
                    metadata=s.metadata,
                    expires_at=s.expires_at,
                    status=s.status,
                )
            )

        return result

    async def export_all(
        self, filename: str = "oauth-seed-data.json", merge: bool = False
    ) -> str:
        """
        Export all seed data to S3.

        Args:
            filename: Output filename in S3.
            merge: If True, merge with existing S3 data (upsert). If False, overwrite.

        Returns:
            S3 URI of the uploaded file.
        """
        logger.info("Starting seed data export (merge=%s)...", merge)

        # Load existing data if merging
        existing_data: Optional[SeedData] = None
        if merge:
            try:
                raw = await self.s3_client.download_seed_data(filename)
                existing_data = SeedData.model_validate(raw)
                logger.info(
                    "Loaded existing seed data: %d users, %d connections, %d resources, %d secrets",
                    len(existing_data.tables.users),
                    len(existing_data.tables.oauth_connections),
                    len(existing_data.tables.integration_resources),
                    len(existing_data.tables.integration_secrets),
                )
            except FileNotFoundError:
                logger.info("No existing seed data found, will create new file")

        # Export current DB state
        new_users = await self.export_users()
        logger.info("Exported %d users from DB", len(new_users))

        new_oauth_connections = await self.export_oauth_connections()
        logger.info("Exported %d OAuth connections from DB", len(new_oauth_connections))

        new_resources = await self.export_integration_resources()
        logger.info("Exported %d integration resources from DB", len(new_resources))

        new_secrets = await self.export_integration_secrets()
        logger.info("Exported %d integration secrets from DB", len(new_secrets))

        # Merge with existing data if requested
        if merge and existing_data:
            users = _merge_by_key(
                [u.model_dump() for u in existing_data.tables.users],
                [u.model_dump() for u in new_users],
                key_fields=["user_id"],
            )
            oauth_connections = _merge_by_key(
                [c.model_dump() for c in existing_data.tables.oauth_connections],
                [c.model_dump() for c in new_oauth_connections],
                key_fields=["user_id", "provider", "provider_account_id"],
            )
            resources = _merge_by_key(
                [r.model_dump() for r in existing_data.tables.integration_resources],
                [r.model_dump() for r in new_resources],
                key_fields=["user_id", "provider", "resource_type", "resource_id"],
            )
            secrets = _merge_by_key(
                [s.model_dump() for s in existing_data.tables.integration_secrets],
                [s.model_dump() for s in new_secrets],
                key_fields=["user_id", "provider", "name"],
            )

            # Convert back to Pydantic models
            final_users = [UserSeed.model_validate(u) for u in users]
            final_oauth = [OAuthConnectionSeed.model_validate(c) for c in oauth_connections]
            final_resources = [IntegrationResourceSeed.model_validate(r) for r in resources]
            final_secrets = [IntegrationSecretSeed.model_validate(s) for s in secrets]

            logger.info(
                "After merge: %d users, %d connections, %d resources, %d secrets",
                len(final_users), len(final_oauth), len(final_resources), len(final_secrets),
            )
        else:
            final_users = new_users
            final_oauth = new_oauth_connections
            final_resources = new_resources
            final_secrets = new_secrets

        seed_data = SeedData(
            version="1.0",
            exported_at=datetime.now(timezone.utc),
            source_environment=config.env,
            tables=SeedTables(
                users=final_users,
                oauth_connections=final_oauth,
                integration_resources=final_resources,
                integration_secrets=final_secrets,
            ),
        )

        s3_path = await self.s3_client.upload_seed_data(
            seed_data.model_dump(mode="json"),
            filename=filename,
        )

        logger.info("Export complete: %s", s3_path)
        return s3_path
