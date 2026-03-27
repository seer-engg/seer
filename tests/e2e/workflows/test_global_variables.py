# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
"""
E2E tests for global variables CRUD and usage.

Tests the complete journey: Create variable → Use in workflow → Update → Verify.
Global variables are org-scoped, so these tests also exercise the organization context.
"""
import pytest


pytestmark = pytest.mark.e2e


class TestGlobalVariableCRUD:
    """Tests for global variable create/read/update/delete."""

    async def test_create_variable(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Creating a global variable should return 201."""
        response = await authenticated_e2e_client.post(
            "/api/v1/variables",
            json={
                "key": "test_api_url",
                "value": "https://api.example.com",
                "is_secret": False,
                "description": "Test API endpoint",
            },
        )
        assert response.status_code == 201, f"Create failed: {response.text}"

        data = response.json()
        assert data["key"] == "test_api_url"
        assert data["value"] == "https://api.example.com"
        assert data["is_secret"] is False
        assert data["description"] == "Test API endpoint"
        assert "id" in data

    async def test_create_secret_variable(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Secret variables should have their value redacted in responses."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/variables",
            json={
                "key": "secret_token",
                "value": "sk-supersecret123",
                "is_secret": True,
            },
        )
        assert create_resp.status_code == 201

        # List variables and check secret is redacted
        list_resp = await authenticated_e2e_client.get("/api/v1/variables")
        assert list_resp.status_code == 200

        items = list_resp.json()["items"]
        secret_var = next((v for v in items if v["key"] == "secret_token"), None)
        assert secret_var is not None
        assert secret_var["is_secret"] is True
        # Value should be empty/redacted for secrets
        assert secret_var["value"] == ""

    async def test_list_variables(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Listing variables should return all created variables."""
        # Create two variables
        await authenticated_e2e_client.post(
            "/api/v1/variables",
            json={"key": "list_var_1", "value": "val1"},
        )
        await authenticated_e2e_client.post(
            "/api/v1/variables",
            json={"key": "list_var_2", "value": "val2"},
        )

        response = await authenticated_e2e_client.get("/api/v1/variables")
        assert response.status_code == 200

        data = response.json()
        assert "items" in data
        keys = {v["key"] for v in data["items"]}
        assert "list_var_1" in keys
        assert "list_var_2" in keys

    async def test_update_variable(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Updating a variable should change its value."""
        # Create
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/variables",
            json={"key": "updatable_var", "value": "original"},
        )
        var_id = create_resp.json()["id"]

        # Update
        update_resp = await authenticated_e2e_client.patch(
            f"/api/v1/variables/{var_id}",
            json={"value": "updated_value"},
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["value"] == "updated_value"

    async def test_delete_variable(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Deleting a variable should remove it from the list."""
        # Create
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/variables",
            json={"key": "deletable_var", "value": "temp"},
        )
        var_id = create_resp.json()["id"]

        # Delete
        delete_resp = await authenticated_e2e_client.delete(
            f"/api/v1/variables/{var_id}"
        )
        assert delete_resp.status_code == 200

        # Verify gone
        list_resp = await authenticated_e2e_client.get("/api/v1/variables")
        keys = {v["key"] for v in list_resp.json()["items"]}
        assert "deletable_var" not in keys

    async def test_duplicate_key_rejected(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Creating a variable with a duplicate key should fail."""
        await authenticated_e2e_client.post(
            "/api/v1/variables",
            json={"key": "unique_key", "value": "first"},
        )

        dup_resp = await authenticated_e2e_client.post(
            "/api/v1/variables",
            json={"key": "unique_key", "value": "second"},
        )
        assert dup_resp.status_code == 409


class TestGlobalVariableEdgeCases:
    """Tests for edge cases in global variables."""

    async def test_update_to_secret(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Changing a variable to secret should redact its value in responses."""
        # Create non-secret
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/variables",
            json={"key": "becomes_secret", "value": "visible"},
        )
        var_id = create_resp.json()["id"]

        # Update to secret
        update_resp = await authenticated_e2e_client.patch(
            f"/api/v1/variables/{var_id}",
            json={"is_secret": True},
        )
        assert update_resp.status_code == 200

        # List and verify redaction
        list_resp = await authenticated_e2e_client.get("/api/v1/variables")
        var = next(v for v in list_resp.json()["items"] if v["key"] == "becomes_secret")
        assert var["is_secret"] is True
        assert var["value"] == ""

    async def test_update_description_only(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Updating only the description should not change value or secret flag."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/variables",
            json={"key": "desc_update", "value": "stable_value"},
        )
        var_id = create_resp.json()["id"]

        update_resp = await authenticated_e2e_client.patch(
            f"/api/v1/variables/{var_id}",
            json={"description": "New description"},
        )
        assert update_resp.status_code == 200

        data = update_resp.json()
        assert data["value"] == "stable_value"
        assert data["description"] == "New description"
        assert data["is_secret"] is False
