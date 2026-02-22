"""
Seed data importer.

Imports User, OAuthConnection, IntegrationResource, and IntegrationSecret
data from S3 JSON into the database using upsert patterns.
"""

from __future__ import annotations

from typing import Optional

from seer.database import User
from seer.database.models_integrations import IntegrationResource, IntegrationSecret
from seer.database.models_oauth import OAuthConnection
from seer.database.seed.s3_client import SeedS3Client
from seer.database.seed.schema import (
    IntegrationResourceRef,
    OAuthConnectionRef,
    SeedData,
)
from seer.logger import get_logger

logger = get_logger("seer.database.seed.importer")


class SeedImporter:
    """Import database seed data from S3."""

    def __init__(self, s3_client: Optional[SeedS3Client] = None):
        """
        Initialize the importer.

        Args:
            s3_client: S3 client for downloads. Created automatically if not provided.
        """
        self.s3_client = s3_client or SeedS3Client()

        # Caches for FK resolution (populated during import)
        self._user_cache: dict[str, User] = {}
        self._oauth_cache: dict[tuple[int, str, str], OAuthConnection] = {}
        self._resource_cache: dict[tuple[int, str, str], IntegrationResource] = {}

    def _clear_caches(self) -> None:
        """Clear all FK resolution caches."""
        self._user_cache.clear()
        self._oauth_cache.clear()
        self._resource_cache.clear()

    async def _resolve_user(self, user_id: str) -> User:
        """
        Resolve user by user_id, using cache.

        Args:
            user_id: The user_id to look up.

        Returns:
            The User instance.

        Raises:
            ValueError: If user not found (should not happen if import order is correct).
        """
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        user = await User.filter(user_id=user_id).first()
        if not user:
            raise ValueError(f"User not found: {user_id}. Ensure users are imported first.")

        self._user_cache[user_id] = user
        return user

    async def _resolve_oauth_connection(
        self, user: User, ref: Optional[OAuthConnectionRef]
    ) -> Optional[OAuthConnection]:
        """
        Resolve OAuth connection by natural key.

        Args:
            user: The User instance.
            ref: OAuth connection reference (provider + account_id).

        Returns:
            The OAuthConnection instance or None if ref is None.
        """
        if ref is None:
            return None

        cache_key = (user.id, ref.provider, ref.provider_account_id)
        if cache_key in self._oauth_cache:
            return self._oauth_cache[cache_key]

        conn = await OAuthConnection.filter(
            user=user,
            provider=ref.provider,
            provider_account_id=ref.provider_account_id,
        ).first()

        if conn:
            self._oauth_cache[cache_key] = conn

        return conn

    async def _resolve_integration_resource(
        self,
        oauth_connection: Optional[OAuthConnection],
        ref: Optional[IntegrationResourceRef],
    ) -> Optional[IntegrationResource]:
        """
        Resolve integration resource by natural key.

        Args:
            oauth_connection: The OAuthConnection instance (required for lookup).
            ref: Resource reference (provider + type + id).

        Returns:
            The IntegrationResource instance or None if ref is None.
        """
        if ref is None or oauth_connection is None:
            return None

        cache_key = (oauth_connection.id, ref.resource_type, ref.resource_id)
        if cache_key in self._resource_cache:
            return self._resource_cache[cache_key]

        resource = await IntegrationResource.filter(
            oauth_connection=oauth_connection,
            resource_type=ref.resource_type,
            resource_id=ref.resource_id,
        ).first()

        if resource:
            self._resource_cache[cache_key] = resource

        return resource

    async def import_users(self, seed_data: SeedData) -> int:
        """
        Import users using get_or_create + update pattern.

        Args:
            seed_data: The parsed seed data.

        Returns:
            Number of users processed.
        """
        count = 0
        for u in seed_data.tables.users:
            user, created = await User.get_or_create(
                user_id=u.user_id,
                defaults={
                    "email": u.email,
                    "first_name": u.first_name,
                    "last_name": u.last_name,
                    "claims": u.claims,
                    "signup_source": u.signup_source,
                    "default_workflow_creation_mode": u.default_workflow_creation_mode,
                },
            )

            if not created:
                # Update existing user (but preserve signup_source)
                user.email = u.email
                user.first_name = u.first_name
                user.last_name = u.last_name
                user.claims = u.claims
                user.default_workflow_creation_mode = u.default_workflow_creation_mode
                await user.save()

            self._user_cache[u.user_id] = user
            count += 1

        return count

    async def import_oauth_connections(self, seed_data: SeedData) -> int:
        """
        Import OAuth connections using update_or_create pattern.

        Args:
            seed_data: The parsed seed data.

        Returns:
            Number of connections processed.
        """
        count = 0
        for c in seed_data.tables.oauth_connections:
            user = await self._resolve_user(c.user_id)

            conn, _ = await OAuthConnection.update_or_create(
                user=user,
                provider=c.provider,
                provider_account_id=c.provider_account_id,
                defaults={
                    "access_token_enc": c.access_token_enc,
                    "refresh_token_enc": c.refresh_token_enc,
                    "expires_at": c.expires_at,
                    "scopes": c.scopes,
                    "token_type": c.token_type,
                    "provider_metadata": c.provider_metadata,
                    "status": c.status,
                },
            )

            # Cache for later FK resolution
            cache_key = (user.id, c.provider, c.provider_account_id)
            self._oauth_cache[cache_key] = conn
            count += 1

        return count

    async def import_integration_resources(self, seed_data: SeedData) -> int:
        """
        Import integration resources using update_or_create pattern.

        Args:
            seed_data: The parsed seed data.

        Returns:
            Number of resources processed.
        """
        count = 0
        for r in seed_data.tables.integration_resources:
            user = await self._resolve_user(r.user_id)
            oauth_conn = await self._resolve_oauth_connection(user, r.oauth_connection_ref)

            resource, _ = await IntegrationResource.update_or_create(
                oauth_connection=oauth_conn,
                resource_type=r.resource_type,
                resource_id=r.resource_id,
                defaults={
                    "user": user,
                    "provider": r.provider,
                    "resource_key": r.resource_key,
                    "name": r.name,
                    "resource_metadata": r.resource_metadata,
                    "status": r.status,
                },
            )

            # Cache for later FK resolution
            if oauth_conn:
                cache_key = (oauth_conn.id, r.resource_type, r.resource_id)
                self._resource_cache[cache_key] = resource

            count += 1

        return count

    async def import_integration_secrets(self, seed_data: SeedData) -> int:
        """
        Import integration secrets using update_or_create pattern.

        Secrets have compound unique constraints:
        - (oauth_connection, name)
        - (resource, name)

        Args:
            seed_data: The parsed seed data.

        Returns:
            Number of secrets processed.
        """
        count = 0
        for s in seed_data.tables.integration_secrets:
            user = await self._resolve_user(s.user_id)
            oauth_conn = await self._resolve_oauth_connection(user, s.oauth_connection_ref)
            resource = await self._resolve_integration_resource(oauth_conn, s.resource_ref)

            # Determine which unique constraint to use
            if oauth_conn:
                # Use (oauth_connection, name) constraint
                await IntegrationSecret.update_or_create(
                    oauth_connection=oauth_conn,
                    name=s.name,
                    defaults={
                        "user": user,
                        "provider": s.provider,
                        "resource": resource,
                        "secret_type": s.secret_type,
                        "value_enc": s.value_enc,
                        "value_fingerprint": s.value_fingerprint,
                        "metadata": s.metadata,
                        "expires_at": s.expires_at,
                        "status": s.status,
                    },
                )
            elif resource:
                # Use (resource, name) constraint
                await IntegrationSecret.update_or_create(
                    resource=resource,
                    name=s.name,
                    defaults={
                        "user": user,
                        "provider": s.provider,
                        "oauth_connection": oauth_conn,
                        "secret_type": s.secret_type,
                        "value_enc": s.value_enc,
                        "value_fingerprint": s.value_fingerprint,
                        "metadata": s.metadata,
                        "expires_at": s.expires_at,
                        "status": s.status,
                    },
                )
            else:
                # Secret without FK - create directly
                # This shouldn't happen in normal data but handle gracefully
                logger.warning(
                    "Secret '%s' has no oauth_connection or resource reference, skipping",
                    s.name,
                )
                continue

            count += 1

        return count

    async def import_all(self, filename: str = "oauth-seed-data.json") -> dict[str, int]:
        """
        Import all seed data from S3.

        Args:
            filename: Filename to download from S3.

        Returns:
            Dictionary with counts of imported records per table.
        """
        logger.info("Starting seed data import from %s...", filename)

        # Clear caches from any previous import
        self._clear_caches()

        # Download and parse
        raw_data = await self.s3_client.download_seed_data(filename)
        seed_data = SeedData.model_validate(raw_data)

        logger.info(
            "Seed data v%s from %s (exported %s)",
            seed_data.version,
            seed_data.source_environment or "unknown",
            seed_data.exported_at,
        )

        # Import in dependency order
        users_count = await self.import_users(seed_data)
        logger.info("Imported %d users", users_count)

        oauth_count = await self.import_oauth_connections(seed_data)
        logger.info("Imported %d OAuth connections", oauth_count)

        resources_count = await self.import_integration_resources(seed_data)
        logger.info("Imported %d integration resources", resources_count)

        secrets_count = await self.import_integration_secrets(seed_data)
        logger.info("Imported %d integration secrets", secrets_count)

        logger.info("Import complete!")

        return {
            "users": users_count,
            "oauth_connections": oauth_count,
            "integration_resources": resources_count,
            "integration_secrets": secrets_count,
        }
