"""Tests for the global variables CRUD API."""
import pytest

from seer.database.workflow_models import GlobalVariable
from seer.database.organization_models import Organization, OrganizationType


async def _make_org(user, slug):
    return await Organization.create(name=f"Org {slug}", slug=slug, type=OrganizationType.TEAM, owner=user)


@pytest.mark.integration
@pytest.mark.asyncio
class TestGlobalVariablesAPI:
    """Integration tests for GlobalVariable model operations."""

    async def test_create_variable(self, db_engine, test_user):
        org = await _make_org(test_user, "test-org")
        var = await GlobalVariable.create(
            organization=org, key="api_key", value="sk-123", is_secret=True, created_by=test_user,
        )
        assert var.id is not None
        assert var.key == "api_key"
        assert var.value == "sk-123"
        assert var.is_secret is True

    async def test_unique_key_per_org(self, db_engine, test_user):
        org = await _make_org(test_user, "org-a")
        await GlobalVariable.create(organization=org, key="foo", value="1", created_by=test_user)
        from tortoise.exceptions import IntegrityError
        with pytest.raises(IntegrityError):
            await GlobalVariable.create(organization=org, key="foo", value="2", created_by=test_user)

    async def test_same_key_different_orgs(self, db_engine, test_user):
        org1 = await _make_org(test_user, "org-1")
        org2 = await _make_org(test_user, "org-2")
        v1 = await GlobalVariable.create(organization=org1, key="shared", value="val1", created_by=test_user)
        v2 = await GlobalVariable.create(organization=org2, key="shared", value="val2", created_by=test_user)
        assert v1.id != v2.id

    async def test_filter_by_org(self, db_engine, test_user):
        org = await _make_org(test_user, "org-f")
        await GlobalVariable.create(organization=org, key="a", value="1", created_by=test_user)
        await GlobalVariable.create(organization=org, key="b", value="2", created_by=test_user)
        rows = await GlobalVariable.filter(organization=org).order_by("key")
        assert len(rows) == 2
        assert rows[0].key == "a"

    async def test_update_variable(self, db_engine, test_user):
        org = await _make_org(test_user, "org-u")
        var = await GlobalVariable.create(organization=org, key="x", value="old", created_by=test_user)
        var.value = "new"
        await var.save()
        refreshed = await GlobalVariable.get(id=var.id)
        assert refreshed.value == "new"

    async def test_delete_variable(self, db_engine, test_user):
        org = await _make_org(test_user, "org-d")
        var = await GlobalVariable.create(organization=org, key="del", value="gone", created_by=test_user)
        deleted = await GlobalVariable.filter(id=var.id).delete()
        assert deleted == 1
        assert await GlobalVariable.filter(id=var.id).first() is None
