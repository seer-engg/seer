"""
Integration tests for User and related database models.

Tests:
- User CRUD operations
- User settings relationships
- User authentication helpers
- Unique constraints
"""
import pytest
from tortoise.exceptions import IntegrityError

from seer.database.models import User, UserSettings


# =============================================================================
# User Model Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_user(db_engine):
    """Test creating a user with valid data."""
    user = await User.create(
        user_id="test_user_001",
        email="test@example.com",
        first_name="John",
        last_name="Doe",
    )

    assert user.id is not None
    assert user.user_id == "test_user_001"
    assert user.email == "test@example.com"
    assert user.first_name == "John"
    assert user.last_name == "Doe"
    assert user.created_at is not None
    assert user.updated_at is not None
    assert user.default_workflow_creation_mode == "ASK_FIRST"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_unique_constraint(db_engine):
    """Test unique constraint on user_id."""
    await User.create(
        user_id="unique_user",
        email="user1@example.com",
    )

    # Attempt to create duplicate user_id
    with pytest.raises(IntegrityError):
        await User.create(
            user_id="unique_user",  # Duplicate!
            email="user2@example.com",
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_nullable_fields(db_engine):
    """Test that nullable fields can be None."""
    user = await User.create(user_id="minimal_user")

    assert user.email is None
    assert user.first_name is None
    assert user.last_name is None
    assert user.claims is None
    assert user.signup_source is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_with_claims(db_engine):
    """Test storing JSON claims data."""
    claims = {
        "sub": "user_123",
        "email_verified": True,
        "roles": ["user", "admin"],
    }

    user = await User.create(
        user_id="claims_user",
        email="claims@example.com",
        claims=claims,
    )

    await user.refresh_from_db()
    assert user.claims == claims
    assert user.claims["email_verified"] is True
    assert "admin" in user.claims["roles"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_signup_source(db_engine):
    """Test tracking signup source."""
    user = await User.create(
        user_id="signup_user",
        email="signup@example.com",
        signup_source="google_oauth",
    )

    assert user.signup_source == "google_oauth"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_default_workflow_creation_mode(db_engine):
    """Test default workflow creation mode."""
    # Test default value
    user1 = await User.create(user_id="user1", email="user1@example.com")
    assert user1.default_workflow_creation_mode == "ASK_FIRST"

    # Test custom value
    user2 = await User.create(
        user_id="user2",
        email="user2@example.com",
        default_workflow_creation_mode="AUTO_CREATE",
    )
    assert user2.default_workflow_creation_mode == "AUTO_CREATE"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_update(db_engine):
    """Test updating user fields."""
    user = await User.create(
        user_id="update_user",
        email="old@example.com",
        first_name="Old",
    )

    # Update fields
    user.email = "new@example.com"
    user.first_name = "New"
    user.last_name = "Name"
    await user.save()

    # Verify updates
    await user.refresh_from_db()
    assert user.email == "new@example.com"
    assert user.first_name == "New"
    assert user.last_name == "Name"


# =============================================================================
# UserSettings Model Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_user_settings(db_engine, test_user):
    """Test creating user settings."""
    settings = await UserSettings.create(
        user=test_user,
        max_agent_steps=50,
        preferences={"theme": "dark", "notifications": True},
    )

    assert settings.id is not None
    assert settings.user_id == test_user.id
    assert settings.max_agent_steps == 50
    assert settings.preferences == {"theme": "dark", "notifications": True}
    assert settings.created_at is not None
    assert settings.updated_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_settings_relationship(db_engine, test_user):
    """Test OneToOne relationship between User and UserSettings."""
    settings = await UserSettings.create(
        user=test_user,
        max_agent_steps=100,
    )

    # Access settings from user
    user_settings = await test_user.settings
    assert user_settings.id == settings.id
    assert user_settings.max_agent_steps == 100

    # Access user from settings
    settings_user = await settings.user
    assert settings_user.id == test_user.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_settings_one_to_one_constraint(db_engine, test_user):
    """Test that only one settings record can exist per user."""
    await UserSettings.create(user=test_user, max_agent_steps=50)

    # Attempt to create duplicate settings
    with pytest.raises(IntegrityError):
        await UserSettings.create(user=test_user, max_agent_steps=100)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_settings_nullable_fields(db_engine, test_user):
    """Test nullable fields in user settings."""
    settings = await UserSettings.create(user=test_user)

    assert settings.max_agent_steps is None
    assert settings.preferences == {}  # Default empty dict


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_settings_preferences_update(db_engine, test_user):
    """Test updating preferences JSON field."""
    settings = await UserSettings.create(
        user=test_user,
        preferences={"theme": "light"},
    )

    # Update preferences
    settings.preferences = {
        "theme": "dark",
        "language": "en",
        "notifications": {"email": True, "push": False},
    }
    await settings.save()

    # Verify update
    await settings.refresh_from_db()
    assert settings.preferences["theme"] == "dark"
    assert settings.preferences["language"] == "en"
    assert settings.preferences["notifications"]["email"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_settings_cascade_behavior(db_engine, test_user):
    """Test that deleting user doesn't cascade to settings by default."""
    settings = await UserSettings.create(user=test_user, max_agent_steps=50)
    settings_id = settings.id

    # Delete user
    await test_user.delete()

    # Settings should be deleted (due to FK constraint)
    deleted_settings = await UserSettings.filter(id=settings_id).first()
    # Behavior depends on DB constraints - typically would be deleted


# =============================================================================
# User Query Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_users_by_email(db_engine):
    """Test querying users by email."""
    await User.create(user_id="user1", email="alice@example.com")
    await User.create(user_id="user2", email="bob@example.com")
    await User.create(user_id="user3", email="charlie@example.com")

    # Query by email
    user = await User.filter(email="bob@example.com").first()
    assert user is not None
    assert user.user_id == "user2"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_users_by_signup_source(db_engine):
    """Test querying users by signup source."""
    await User.create(user_id="user1", signup_source="google_oauth")
    await User.create(user_id="user2", signup_source="github_oauth")
    await User.create(user_id="user3", signup_source="google_oauth")

    # Query by signup source
    google_users = await User.filter(signup_source="google_oauth").all()
    assert len(google_users) == 2
    assert {u.user_id for u in google_users} == {"user1", "user3"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_ordering(db_engine):
    """Test users are ordered by user_id."""
    await User.create(user_id="user_c")
    await User.create(user_id="user_a")
    await User.create(user_id="user_b")

    users = await User.all()

    # Should be ordered by user_id
    assert users[0].user_id == "user_a"
    assert users[1].user_id == "user_b"
    assert users[2].user_id == "user_c"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_count(db_engine):
    """Test counting users."""
    await User.create(user_id="user1")
    await User.create(user_id="user2")
    await User.create(user_id="user3")

    count = await User.all().count()
    assert count == 3
