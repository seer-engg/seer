"""
Integration tests for OAuth connection and resource binding lifecycle.

Tests connection CRUD, cascade deletion, resource binding (Supabase/Postgres),
secret management, and multi-account isolation with real database.

Only external services (OAuth providers, HTTP) are mocked.
"""
import pytest
from fastapi import HTTPException

from seer.api.integrations.services import (
    bind_postgres_database_manual,
    bind_supabase_project_manual,
    deactivate_integration_resource,
    delete_connection_by_id,
    disconnect_provider,
    get_connection_for_provider,
    list_integration_resources,
    list_postgres_bindings,
    list_resource_secrets,
    _parse_postgres_connection_string,
)
from seer.database import IntegrationResource, IntegrationSecret, OAuthConnection, User


# =============================================================================
# Helpers
# =============================================================================


async def _create_oauth_connection(user, provider="google", account_id="user@gmail.com", **kw):
    """Create a real OAuthConnection in the database."""
    defaults = dict(
        user=user,
        provider=provider,
        provider_account_id=account_id,
        access_token_enc="token_123",
        status="active",
    )
    defaults.update(kw)
    return await OAuthConnection.create(**defaults)


async def _create_resource(user, connection=None, provider="supabase", **kw):
    """Create a real IntegrationResource in the database."""
    defaults = dict(
        user=user,
        oauth_connection=connection,
        provider=provider,
        resource_type="project",
        resource_id="proj_abc",
        resource_key="proj_abc",
        name="Test Project",
        status="active",
    )
    defaults.update(kw)
    return await IntegrationResource.create(**defaults)


async def _create_secret(user, resource=None, connection=None, provider="supabase", **kw):
    """Create a real IntegrationSecret in the database."""
    defaults = dict(
        user=user,
        provider=provider,
        oauth_connection=connection,
        resource=resource,
        secret_type="api_key",
        name="test_key",
        value_enc="encrypted_value",
        status="active",
    )
    defaults.update(kw)
    return await IntegrationSecret.create(**defaults)


# =============================================================================
# Connection Lifecycle
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestConnectionLifecycle:
    """OAuth connection create, lookup, and delete with real DB."""

    async def test_get_connection_for_provider(self, db_engine, test_user):
        """Active connection is found by provider name."""
        conn = await _create_oauth_connection(test_user)

        result = await get_connection_for_provider(test_user, "google")
        assert result is not None
        assert result.id == conn.id

    async def test_get_connection_returns_none_for_revoked(self, db_engine, test_user):
        """Revoked connection is not returned."""
        await _create_oauth_connection(test_user, status="revoked")

        result = await get_connection_for_provider(test_user, "google")
        assert result is None

    async def test_get_connection_returns_none_for_other_provider(self, db_engine, test_user):
        """Connection for different provider is not returned."""
        await _create_oauth_connection(test_user, provider="github", account_id="user")

        result = await get_connection_for_provider(test_user, "google")
        assert result is None

    async def test_delete_connection_by_id(self, db_engine, test_user):
        """Delete connection by ID → status becomes 'revoked'."""
        conn = await _create_oauth_connection(test_user)

        await delete_connection_by_id(test_user, str(conn.id))

        await conn.refresh_from_db()
        assert conn.status == "revoked"

    async def test_delete_connection_with_prefixed_id(self, db_engine, test_user):
        """Delete connection with 'provider:id' format works."""
        conn = await _create_oauth_connection(test_user)

        await delete_connection_by_id(test_user, f"google:{conn.id}")

        await conn.refresh_from_db()
        assert conn.status == "revoked"

    async def test_delete_nonexistent_connection_raises_404(self, db_engine, test_user):
        """Deleting nonexistent connection raises 404."""
        with pytest.raises(HTTPException) as exc_info:
            await delete_connection_by_id(test_user, "99999")
        assert exc_info.value.status_code == 404

    async def test_get_connection_multiple_accounts_no_crash(self, db_engine, test_user):
        """Multiple active connections for same provider returns most recent, not crash."""
        conn1 = await _create_oauth_connection(test_user, account_id="acct1@gmail.com")
        conn2 = await _create_oauth_connection(test_user, account_id="acct2@gmail.com")
        # Touch conn2 so it has a more recent updated_at
        conn2.scopes = "openid"
        await conn2.save()

        result = await get_connection_for_provider(test_user, "google")
        assert result is not None
        assert result.id == conn2.id

    async def test_get_connection_with_connection_id(self, db_engine, test_user):
        """When connection_id is provided, returns that specific connection."""
        conn1 = await _create_oauth_connection(test_user, account_id="acct1@gmail.com")
        conn2 = await _create_oauth_connection(test_user, account_id="acct2@gmail.com")

        result = await get_connection_for_provider(
            test_user, "google", connection_id=conn1.id
        )
        assert result is not None
        assert result.id == conn1.id

    async def test_get_connection_with_invalid_connection_id(self, db_engine, test_user):
        """Non-existent connection_id returns None."""
        await _create_oauth_connection(test_user)

        result = await get_connection_for_provider(
            test_user, "google", connection_id=99999
        )
        assert result is None

    async def test_get_connection_with_wrong_provider_connection_id(self, db_engine, test_user):
        """connection_id for different provider returns None (provider guard)."""
        github_conn = await _create_oauth_connection(
            test_user, provider="github", account_id="ghuser"
        )

        result = await get_connection_for_provider(
            test_user, "google", connection_id=github_conn.id
        )
        assert result is None


# =============================================================================
# Cascade Deletion
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestCascadeDeletion:
    """Disconnecting a provider cascades to resources and secrets."""

    async def test_disconnect_provider_revokes_connection(self, db_engine, test_user):
        """disconnect_provider sets connection status to 'revoked'."""
        conn = await _create_oauth_connection(test_user)

        await disconnect_provider(test_user, "google")

        await conn.refresh_from_db()
        assert conn.status == "revoked"

    async def test_disconnect_cascades_to_resources(self, db_engine, test_user):
        """Resources linked to revoked connection are also revoked."""
        conn = await _create_oauth_connection(test_user)
        resource = await _create_resource(test_user, connection=conn)

        await disconnect_provider(test_user, "google")

        await resource.refresh_from_db()
        assert resource.status == "revoked"

    async def test_disconnect_cascades_to_secrets(self, db_engine, test_user):
        """Secrets linked to revoked connection are also revoked."""
        conn = await _create_oauth_connection(test_user)
        resource = await _create_resource(test_user, connection=conn)
        secret = await _create_secret(test_user, resource=resource, connection=conn)

        await disconnect_provider(test_user, "google")

        await secret.refresh_from_db()
        assert secret.status == "revoked"

    async def test_disconnect_cascades_connection_level_secrets(self, db_engine, test_user):
        """Secrets directly on connection (no resource) are also revoked."""
        conn = await _create_oauth_connection(test_user)
        secret = await _create_secret(
            test_user, connection=conn, resource=None, name="conn_secret"
        )

        await disconnect_provider(test_user, "google")

        await secret.refresh_from_db()
        assert secret.status == "revoked"

    async def test_disconnect_revokes_all_connections_for_provider(self, db_engine, test_user):
        """Multiple connections for same provider are all revoked."""
        conn1 = await _create_oauth_connection(test_user, account_id="acct1@gmail.com")
        conn2 = await _create_oauth_connection(test_user, account_id="acct2@gmail.com")

        await disconnect_provider(test_user, "google")

        await conn1.refresh_from_db()
        await conn2.refresh_from_db()
        assert conn1.status == "revoked"
        assert conn2.status == "revoked"

    async def test_disconnect_does_not_affect_other_providers(self, db_engine, test_user):
        """Disconnecting Google doesn't affect GitHub connections."""
        google_conn = await _create_oauth_connection(test_user, provider="google", account_id="g@gmail.com")
        github_conn = await _create_oauth_connection(test_user, provider="github", account_id="ghuser")

        await disconnect_provider(test_user, "google")

        await google_conn.refresh_from_db()
        await github_conn.refresh_from_db()
        assert google_conn.status == "revoked"
        assert github_conn.status == "active"

    async def test_delete_connection_cascades_to_resources_and_secrets(self, db_engine, test_user):
        """Deleting a single connection cascades to its resources and secrets."""
        conn = await _create_oauth_connection(test_user)
        resource = await _create_resource(test_user, connection=conn)
        secret = await _create_secret(test_user, resource=resource, connection=conn)

        await delete_connection_by_id(test_user, str(conn.id))

        await resource.refresh_from_db()
        await secret.refresh_from_db()
        assert resource.status == "revoked"
        assert secret.status == "revoked"


# =============================================================================
# Resource Binding — Supabase Manual
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestSupabaseManualBinding:
    """Supabase manual binding creates resource + secrets in DB."""

    async def test_bind_creates_resource_and_service_key(self, db_engine, test_user):
        """Manual Supabase bind creates resource + service_role_key secret."""
        resource = await bind_supabase_project_manual(
            test_user,
            project_ref="my-project",
            service_role_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test",
            project_name="My Project",
        )

        assert resource.provider == "supabase"
        assert resource.resource_id == "my-project"
        assert resource.name == "My Project"
        assert resource.resource_metadata["project_ref"] == "my-project"
        assert "rest_url" in resource.resource_metadata

        # Secret created
        secrets = await IntegrationSecret.filter(resource=resource, status="active").all()
        assert len(secrets) == 1
        assert secrets[0].name == "supabase_service_role_key"
        assert secrets[0].value_fingerprint is not None

    async def test_bind_with_anon_key_creates_two_secrets(self, db_engine, test_user):
        """Providing anon_key creates both service_role and anon secrets."""
        resource = await bind_supabase_project_manual(
            test_user,
            project_ref="dual-keys",
            service_role_key="service_key_value",
            anon_key="anon_key_value",
        )

        secrets = await IntegrationSecret.filter(resource=resource, status="active").all()
        secret_names = {s.name for s in secrets}
        assert secret_names == {"supabase_service_role_key", "supabase_anon_key"}

    async def test_bind_upserts_on_duplicate(self, db_engine, test_user):
        """Binding same project_ref twice upserts instead of creating duplicate."""
        r1 = await bind_supabase_project_manual(
            test_user, project_ref="upsert-test", service_role_key="key_v1"
        )
        r2 = await bind_supabase_project_manual(
            test_user, project_ref="upsert-test", service_role_key="key_v2"
        )

        assert r1.id == r2.id  # Same resource
        secret = await IntegrationSecret.get(resource=r2, name="supabase_service_role_key")
        assert secret.value_enc == "key_v2"  # Updated

    async def test_bind_missing_service_key_raises(self, db_engine, test_user):
        """Missing service_role_key raises 400."""
        with pytest.raises(HTTPException) as exc_info:
            await bind_supabase_project_manual(
                test_user, project_ref="no-key", service_role_key=""
            )
        assert exc_info.value.status_code == 400


# =============================================================================
# Resource Binding — Postgres Manual
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestPostgresManualBinding:
    """Postgres manual binding creates resource + connection string secret."""

    async def test_bind_with_individual_fields(self, db_engine, test_user):
        """Binding with host/port/db/user/pass creates resource + secret."""
        resource = await bind_postgres_database_manual(
            test_user,
            name="My Database",
            host="db.example.com",
            port=5432,
            database="mydb",
            db_user="admin",
            password="secret123",
        )

        assert resource.provider == "postgres"
        assert resource.name == "My Database"
        assert resource.resource_metadata["host"] == "db.example.com"
        assert resource.resource_metadata["access_mode"] == "restricted"

        secrets = await IntegrationSecret.filter(resource=resource, status="active").all()
        assert len(secrets) == 1
        assert secrets[0].name == "postgres_connection_string"
        assert "secret123" in secrets[0].value_enc  # Connection string contains password

    async def test_bind_with_connection_string(self, db_engine, test_user):
        """Binding with connection string parses and stores it."""
        resource = await bind_postgres_database_manual(
            test_user,
            name="Parsed DB",
            connection_string="postgresql://admin:pass@db.host.com:5433/proddb?sslmode=require",
        )

        assert resource.resource_metadata["host"] == "db.host.com"
        assert resource.resource_metadata["port"] == 5433
        assert resource.resource_metadata["database"] == "proddb"
        assert resource.resource_metadata["ssl_mode"] == "require"

    async def test_bind_missing_host_raises(self, db_engine, test_user):
        """Missing host without connection string raises 400."""
        with pytest.raises(HTTPException) as exc_info:
            await bind_postgres_database_manual(
                test_user, name="Bad", database="db", db_user="u", password="p"
            )
        assert exc_info.value.status_code == 400

    async def test_list_postgres_bindings(self, db_engine, test_user):
        """list_postgres_bindings returns only postgres resources."""
        await bind_postgres_database_manual(
            test_user, name="PG1", host="h1", database="d1", db_user="u", password="p"
        )
        await bind_supabase_project_manual(
            test_user, project_ref="sb1", service_role_key="k"
        )

        pg_bindings = await list_postgres_bindings(test_user)
        assert len(pg_bindings) == 1
        assert pg_bindings[0].name == "PG1"


# =============================================================================
# Resource & Secret Management
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestResourceManagement:
    """Resource listing, filtering, deactivation, and secret listing."""

    async def test_list_resources_by_provider(self, db_engine, test_user):
        """list_integration_resources filters by provider."""
        await _create_resource(test_user, provider="supabase", resource_id="sb1")
        await _create_resource(test_user, provider="postgres", resource_id="pg1", resource_type="database")

        sb_resources = await list_integration_resources(test_user, provider="supabase")
        assert len(sb_resources) == 1
        assert sb_resources[0].provider == "supabase"

    async def test_list_resources_excludes_revoked(self, db_engine, test_user):
        """Revoked resources are not returned."""
        active = await _create_resource(test_user, resource_id="active_r")
        revoked = await _create_resource(test_user, resource_id="revoked_r")
        revoked.status = "revoked"
        await revoked.save()

        resources = await list_integration_resources(test_user)
        ids = {r.resource_id for r in resources}
        assert "active_r" in ids
        assert "revoked_r" not in ids

    async def test_deactivate_resource_cascades_to_secrets(self, db_engine, test_user):
        """Deactivating a resource also revokes its secrets."""
        resource = await _create_resource(test_user, resource_id="deact_r")
        secret = await _create_secret(test_user, resource=resource, name="deact_secret")

        await deactivate_integration_resource(test_user, resource.id)

        await resource.refresh_from_db()
        await secret.refresh_from_db()
        assert resource.status == "revoked"
        assert secret.status == "revoked"

    async def test_deactivate_nonexistent_raises_404(self, db_engine, test_user):
        """Deactivating nonexistent resource raises 404."""
        with pytest.raises(HTTPException) as exc_info:
            await deactivate_integration_resource(test_user, 99999)
        assert exc_info.value.status_code == 404

    async def test_list_resource_secrets(self, db_engine, test_user):
        """list_resource_secrets returns only active secrets for the resource."""
        resource = await _create_resource(test_user, resource_id="sec_list_r")
        s1 = await _create_secret(test_user, resource=resource, name="key_a")
        s2 = await _create_secret(test_user, resource=resource, name="key_b")
        s3 = await _create_secret(test_user, resource=resource, name="key_revoked")
        s3.status = "revoked"
        await s3.save()

        secrets = await list_resource_secrets(test_user, resource.id)
        names = {s.name for s in secrets}
        assert names == {"key_a", "key_b"}

    async def test_list_resource_secrets_nonexistent_raises_404(self, db_engine, test_user):
        """Listing secrets for nonexistent resource raises 404."""
        with pytest.raises(HTTPException) as exc_info:
            await list_resource_secrets(test_user, 99999)
        assert exc_info.value.status_code == 404


# =============================================================================
# Connection String Parsing (Pure Logic)
# =============================================================================


@pytest.mark.integration
class TestPostgresConnectionStringParsing:
    """_parse_postgres_connection_string — pure logic, no DB."""

    def test_standard_format(self):
        result = _parse_postgres_connection_string(
            "postgresql://admin:pass@db.host.com:5433/mydb?sslmode=require"
        )
        assert result["host"] == "db.host.com"
        assert result["port"] == 5433
        assert result["database"] == "mydb"
        assert result["user"] == "admin"
        assert result["password"] == "pass"
        assert result["ssl_mode"] == "require"

    def test_postgres_scheme(self):
        """postgres:// scheme (without 'ql') is also valid."""
        result = _parse_postgres_connection_string("postgres://u:p@host/db")
        assert result["host"] == "host"
        assert result["user"] == "u"

    def test_default_port(self):
        result = _parse_postgres_connection_string("postgresql://u:p@host/db")
        assert result["port"] == 5432

    def test_default_sslmode(self):
        result = _parse_postgres_connection_string("postgresql://u:p@host/db")
        assert result["ssl_mode"] == "prefer"

    def test_url_encoded_password(self):
        """Passwords with special characters are URL-decoded."""
        result = _parse_postgres_connection_string("postgresql://u:p%40ss%23word@host/db")
        assert result["password"] == "p@ss#word"

    def test_invalid_scheme_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_postgres_connection_string("mysql://u:p@host/db")
        assert exc_info.value.status_code == 400

    def test_missing_database_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_postgres_connection_string("postgresql://u:p@host/")
        assert exc_info.value.status_code == 400

    def test_missing_user_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_postgres_connection_string("postgresql://@host/db")
        assert exc_info.value.status_code == 400


# =============================================================================
# User Isolation
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestUserIsolation:
    """Resources and connections are isolated per user."""

    async def test_connections_isolated_by_user(self, db_engine, test_user):
        """User cannot see another user's connections."""
        user2 = await User.create(
            user_id="integration_user_2", email="int2@example.com",
            first_name="Other", last_name="User",
        )

        await _create_oauth_connection(test_user, account_id="user1@gmail.com")
        await _create_oauth_connection(user2, account_id="user2@gmail.com")

        u1_conn = await get_connection_for_provider(test_user, "google")
        u2_conn = await get_connection_for_provider(user2, "google")

        assert u1_conn.provider_account_id == "user1@gmail.com"
        assert u2_conn.provider_account_id == "user2@gmail.com"

    async def test_resources_isolated_by_user(self, db_engine, test_user):
        """User cannot list another user's resources."""
        user2 = await User.create(
            user_id="integration_user_3", email="int3@example.com",
            first_name="Other", last_name="User",
        )

        await _create_resource(test_user, resource_id="r_user1")
        await _create_resource(user2, resource_id="r_user2")

        u1_resources = await list_integration_resources(test_user)
        u2_resources = await list_integration_resources(user2)

        assert all(r.resource_id != "r_user2" for r in u1_resources)
        assert all(r.resource_id != "r_user1" for r in u2_resources)

    async def test_disconnect_does_not_affect_other_users(self, db_engine, test_user):
        """Disconnecting one user's provider doesn't affect another user."""
        user2 = await User.create(
            user_id="integration_user_4", email="int4@example.com",
            first_name="Other", last_name="User",
        )

        await _create_oauth_connection(test_user, account_id="u1@gmail.com")
        u2_conn = await _create_oauth_connection(user2, account_id="u2@gmail.com")

        await disconnect_provider(test_user, "google")

        await u2_conn.refresh_from_db()
        assert u2_conn.status == "active"
