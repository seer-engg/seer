"""
Integration tests for organization lifecycle service.

Tests with real database: creation, slug generation, conversion, subscriptions.
"""
import pytest

from seer.database import Organization, User
from seer.database.organization_models import (
    MembershipStatus,
    OrganizationRole,
    OrganizationType,
)
from seer.database.subscription_models import (
    BillingSubscription,
    SubscriptionStatus,
    SubscriptionTier,
)
from seer.services.organization_service import (
    convert_personal_to_team,
    create_personal_organization,
    create_team_organization,
    generate_slug,
)


# =============================================================================
# Slug Generation (Pure Logic)
# =============================================================================


@pytest.mark.integration
class TestGenerateSlug:
    """Tests for slug generation."""

    def test_basic_slug(self):
        assert generate_slug("My Workspace") == "my-workspace"

    def test_special_characters_replaced(self):
        assert generate_slug("Test & More!") == "test-more"

    def test_multiple_hyphens_collapsed(self):
        slug = generate_slug("test---multiple---hyphens")
        assert "--" not in slug

    def test_leading_trailing_hyphens_stripped(self):
        slug = generate_slug("--test--")
        assert not slug.startswith("-")
        assert not slug.endswith("-")

    def test_with_suffix(self):
        slug = generate_slug("test", suffix="abc123")
        assert slug == "test-abc123"

    def test_max_length(self):
        slug = generate_slug("x" * 300)
        assert len(slug) <= 255


# =============================================================================
# Personal Organization Creation
# =============================================================================


@pytest.mark.integration
class TestCreatePersonalOrganization:
    """Tests for personal organization creation."""

    @pytest.mark.asyncio
    async def test_creates_org_with_user_name(self, db_engine, test_user):
        """Should create org named after the user."""
        org, membership = await create_personal_organization(test_user)

        assert org.type == OrganizationType.PERSONAL
        assert org.owner_id == test_user.id
        assert test_user.first_name in org.name

    @pytest.mark.asyncio
    async def test_creates_owner_membership(self, db_engine, test_user):
        """Should create OWNER membership for the user."""
        org, membership = await create_personal_organization(test_user)

        assert membership.role == OrganizationRole.OWNER
        assert membership.status == MembershipStatus.ACTIVE
        assert membership.user_id == test_user.id
        assert membership.organization_id == org.id

    @pytest.mark.asyncio
    async def test_creates_free_subscription(self, db_engine, test_user):
        """Should create FREE tier subscription for the org."""
        org, _ = await create_personal_organization(test_user)

        subscription = await BillingSubscription.get(organization=org)
        assert subscription.tier == SubscriptionTier.FREE
        assert subscription.status == SubscriptionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_slug_derived_from_user_id(self, db_engine, test_user):
        """Slug should be derived from user_id for uniqueness."""
        org, _ = await create_personal_organization(test_user)

        assert "personal" in org.slug
        # user_id underscores become hyphens in slug
        assert test_user.user_id.replace("_", "-") in org.slug

    @pytest.mark.asyncio
    async def test_email_fallback_when_no_name(self, db_engine):
        """Should use email prefix when user has no name."""
        user = await User.create(
            user_id="user_no_name",
            email="noname@example.com",
            first_name="",
            last_name="",
        )
        org, _ = await create_personal_organization(user)

        assert "noname" in org.name


# =============================================================================
# Team Organization Creation
# =============================================================================


@pytest.mark.integration
class TestCreateTeamOrganization:
    """Tests for team organization creation."""

    @pytest.mark.asyncio
    async def test_creates_team_org(self, db_engine, test_user):
        """Should create a TEAM type organization."""
        org, membership = await create_team_organization(test_user, "My Team")

        assert org.type == OrganizationType.TEAM
        assert org.name == "My Team"
        assert org.owner_id == test_user.id

    @pytest.mark.asyncio
    async def test_auto_generates_slug(self, db_engine, test_user):
        """Should generate slug from team name."""
        org, _ = await create_team_organization(test_user, "Engineering Team")

        assert org.slug == "engineering-team"

    @pytest.mark.asyncio
    async def test_custom_slug(self, db_engine, test_user):
        """Should use custom slug when provided."""
        org, _ = await create_team_organization(
            test_user, "My Team", slug="custom-slug",
        )

        assert org.slug == "custom-slug"

    @pytest.mark.asyncio
    async def test_duplicate_slug_gets_suffix(self, db_engine, test_user):
        """Duplicate auto-generated slug should get a random suffix."""
        org1, _ = await create_team_organization(test_user, "Same Name")
        org2, _ = await create_team_organization(test_user, "Same Name")

        assert org1.slug != org2.slug
        assert org2.slug.startswith("same-name")

    @pytest.mark.asyncio
    async def test_explicit_duplicate_slug_raises(self, db_engine, test_user):
        """Explicit duplicate slug should raise ValueError."""
        await create_team_organization(test_user, "First", slug="taken-slug")

        with pytest.raises(ValueError, match="already exists"):
            await create_team_organization(test_user, "Second", slug="taken-slug")

    @pytest.mark.asyncio
    async def test_creates_free_subscription(self, db_engine, test_user):
        """Team should get FREE subscription by default."""
        org, _ = await create_team_organization(test_user, "Billing Test Team")

        subscription = await BillingSubscription.get(organization=org)
        assert subscription.tier == SubscriptionTier.FREE


# =============================================================================
# Organization Conversion
# =============================================================================


@pytest.mark.integration
class TestConvertPersonalToTeam:
    """Tests for converting personal → team organization."""

    @pytest.mark.asyncio
    async def test_converts_type(self, db_engine, test_user):
        """Should change organization type from PERSONAL to TEAM."""
        org, _ = await create_personal_organization(test_user)
        assert org.type == OrganizationType.PERSONAL

        converted = await convert_personal_to_team(org, "New Team Name")

        assert converted.type == OrganizationType.TEAM
        assert converted.name == "New Team Name"

    @pytest.mark.asyncio
    async def test_rejects_already_team(self, db_engine, test_user):
        """Should raise ValueError if already a team."""
        org, _ = await create_team_organization(test_user, "Already Team")

        with pytest.raises(ValueError):
            await convert_personal_to_team(org, "New Name")
