"""
SQLModel models - clean slate conversion from Tortoise.

All models use SQLModel which integrates with:
- Pydantic (validation)
- SQLAlchemy (database)
- FastAPI (automatic API schemas)
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, ClassVar, TYPE_CHECKING
from sqlmodel import Field, SQLModel, Relationship, Column
from sqlalchemy import Index, UniqueConstraint, String, Integer, BigInteger, Text
from sqlalchemy.dialects.postgresql import JSONB

if TYPE_CHECKING:
    from api.middleware.auth import AuthenticatedUser


# ====================
# Enums
# ====================

class WorkflowRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowRunSource(str, Enum):
    MANUAL = "manual"
    TRIGGER = "trigger"


class WorkflowVersionStatus(str, Enum):
    DRAFT = "DRAFT"
    RELEASED = "RELEASED"
    ARCHIVED = "ARCHIVED"


class TriggerEventStatus(str, Enum):
    RECEIVED = "received"
    ROUTED = "routed"
    PROCESSED = "processed"
    FAILED = "failed"


# ====================
# Helper Functions
# ====================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


WORKFLOW_ID_PREFIX = "wf_"
RUN_ID_PREFIX = "run_"


def make_workflow_public_id(pk: int) -> str:
    return f"{WORKFLOW_ID_PREFIX}{pk}"


def parse_workflow_public_id(value: str) -> int:
    if not value.startswith(WORKFLOW_ID_PREFIX):
        raise ValueError("Invalid workflow_id format")
    return int(value.removeprefix(WORKFLOW_ID_PREFIX))


def make_run_public_id(pk: int) -> str:
    return f"{RUN_ID_PREFIX}{pk}"


def parse_run_public_id(value: str) -> int:
    if not value.startswith(RUN_ID_PREFIX):
        raise ValueError("Invalid run_id format")
    return int(value.removeprefix(RUN_ID_PREFIX))


# ====================
# Core Models
# ====================

class User(SQLModel, table=True):
    """Database model for authenticated users."""
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(unique=True, index=True, max_length=255)  # Clerk user ID
    email: Optional[str] = Field(default=None, max_length=320)
    first_name: Optional[str] = Field(default=None, max_length=255)
    last_name: Optional[str] = Field(default=None, max_length=255)
    claims: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    signup_source: Optional[str] = Field(default=None, max_length=50)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    workflows: list["Workflow"] = Relationship(back_populates="user")
    workflow_records: list["WorkflowRecord"] = Relationship(back_populates="user")
    workflow_runs: list["WorkflowRun"] = Relationship(back_populates="user")
    chat_sessions: list["WorkflowChatSession"] = Relationship(back_populates="user")
    workflow_proposals: list["WorkflowProposal"] = Relationship(back_populates="created_by")
    trigger_subscriptions: list["TriggerSubscription"] = Relationship(back_populates="user")
    oauth_connections: list["OAuthConnection"] = Relationship(back_populates="user")
    integration_resources: list["IntegrationResource"] = Relationship(back_populates="user")
    integration_secrets: list["IntegrationSecret"] = Relationship(back_populates="user")
    projects: list["Project"] = Relationship(back_populates="user", sa_relationship_kwargs={"foreign_keys": "Project.user_id"})

    @staticmethod
    async def get_or_create_from_auth(auth_user: "AuthenticatedUser", signup_source: Optional[str] = None) -> "User":
        """Fetch or persist a user based on Clerk claims."""
        from sqlmodel import select
        from shared.database.base import async_session_maker

        async with async_session_maker() as session:
            # Try to find existing user
            statement = select(User).where(User.user_id == auth_user.user_id)
            result = await session.execute(statement)
            user = result.scalar_one_or_none()

            if user:
                # Update existing user
                updated_fields = []
                if user.email != auth_user.email:
                    user.email = auth_user.email
                    updated_fields.append('email')
                if user.first_name != auth_user.first_name:
                    user.first_name = auth_user.first_name
                    updated_fields.append('first_name')
                if user.last_name != auth_user.last_name:
                    user.last_name = auth_user.last_name
                    updated_fields.append('last_name')
                if user.claims != auth_user.claims:
                    user.claims = auth_user.claims
                    updated_fields.append('claims')

                if updated_fields:
                    user.updated_at = utc_now()
                    session.add(user)
                    await session.commit()
                    await session.refresh(user)
            else:
                # Create new user
                user = User(
                    user_id=auth_user.user_id,
                    email=auth_user.email,
                    first_name=auth_user.first_name,
                    last_name=auth_user.last_name,
                    claims=auth_user.claims,
                    signup_source=signup_source,
                )
                session.add(user)
                await session.commit()
                await session.refresh(user)

            return user


class Project(SQLModel, table=True):
    """Database model for projects."""
    __tablename__ = "projects"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    project_name: str = Field(unique=True, max_length=255)
    description: Optional[str] = Field(default=None, sa_column=Column(Text))
    project_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSONB, name="metadata"))
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    user: Optional[User] = Relationship(back_populates="projects", sa_relationship_kwargs={"foreign_keys": "Project.user_id"})


class Workflow(SQLModel, table=True):
    """Workflow entity - owns drafts, versions, and published state."""
    __tablename__ = "workflows"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    name: str = Field(max_length=255)
    description: Optional[str] = Field(default=None, sa_column=Column(Text))
    tags: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    meta: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    published_version_id: Optional[int] = Field(
        default=None,
        foreign_key="workflow_versions.id"
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now, index=True)

    # Relationships
    user: User = Relationship(back_populates="workflows")
    versions: list["WorkflowVersion"] = Relationship(back_populates="workflow")
    draft: Optional["WorkflowDraft"] = Relationship(back_populates="workflow", sa_relationship_kwargs={"uselist": False})
    runs: list["WorkflowRun"] = Relationship(back_populates="workflow")
    chat_sessions: list["WorkflowChatSession"] = Relationship(back_populates="workflow")
    proposals: list["WorkflowProposal"] = Relationship(back_populates="workflow")
    trigger_subscriptions: list["TriggerSubscription"] = Relationship(back_populates="workflow")

    @property
    def workflow_id(self) -> str:
        return make_workflow_public_id(self.id)


class WorkflowVersion(SQLModel, table=True):
    """Immutable runnable workflow version."""
    __tablename__ = "workflow_versions"
    __table_args__ = (
        UniqueConstraint('workflow_id', 'version_number', name='uq_workflow_version'),
        Index('ix_workflow_versions_workflow_id', 'workflow_id'),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_id: int = Field(foreign_key="workflows.id", index=True)
    status: WorkflowVersionStatus = Field(default=WorkflowVersionStatus.DRAFT, max_length=20)
    spec: dict = Field(sa_column=Column(JSONB))
    created_from_draft_revision: Optional[int] = None
    created_at: datetime = Field(default_factory=utc_now)
    created_by_id: Optional[int] = Field(default=None, foreign_key="users.id")
    manifest: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    spec_hash: Optional[str] = Field(default=None, max_length=64)
    version_number: Optional[int] = None

    # Relationships
    workflow: Workflow = Relationship(back_populates="versions")
    runs: list["WorkflowRun"] = Relationship(back_populates="workflow_version")

    @property
    def workflow_public_id(self) -> str:
        return make_workflow_public_id(self.workflow_id)


class WorkflowDraft(SQLModel, table=True):
    """Mutable draft state for a workflow."""
    __tablename__ = "workflow_drafts"

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_id: int = Field(foreign_key="workflows.id", unique=True, index=True)
    spec: dict = Field(sa_column=Column(JSONB))
    revision: int = Field(default=1)
    updated_at: datetime = Field(default_factory=utc_now)
    updated_by_id: Optional[int] = Field(default=None, foreign_key="users.id")
    validation_errors: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    validation_warnings: Optional[dict] = Field(default=None, sa_column=Column(JSONB))

    # Relationships
    workflow: Workflow = Relationship(back_populates="draft")

    @property
    def workflow_public_id(self) -> str:
        return make_workflow_public_id(self.workflow_id)


class WorkflowRun(SQLModel, table=True):
    """Persisted workflow run metadata."""
    __tablename__ = "workflow_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    workflow_id: Optional[int] = Field(default=None, foreign_key="workflows.id", index=True)
    workflow_version_id: Optional[int] = Field(
        default=None,
        foreign_key="workflow_versions.id",
        index=True
    )
    spec: dict = Field(sa_column=Column(JSONB))
    inputs: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    config: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    source: WorkflowRunSource = Field(default=WorkflowRunSource.MANUAL, max_length=20)
    subscription_id: Optional[int] = Field(
        default=None,
        foreign_key="trigger_subscriptions.id"
    )
    trigger_event_id: Optional[int] = Field(
        default=None,
        foreign_key="trigger_events.id"
    )
    status: WorkflowRunStatus = Field(default=WorkflowRunStatus.QUEUED, max_length=20)
    output: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    error: Optional[str] = Field(default=None, sa_column=Column(Text))
    thread_id: Optional[str] = Field(default=None, max_length=255)
    created_at: datetime = Field(default_factory=utc_now, index=True)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    metrics: Optional[dict] = Field(default=None, sa_column=Column(JSONB))

    # Relationships
    user: User = Relationship(back_populates="workflow_runs")
    workflow: Optional[Workflow] = Relationship(back_populates="runs")
    workflow_version: Optional[WorkflowVersion] = Relationship(back_populates="runs")

    @property
    def run_id(self) -> str:
        return make_run_public_id(self.id)


class WorkflowChatSession(SQLModel, table=True):
    """Chat session for workflow assistant."""
    __tablename__ = "workflow_chat_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_id: int = Field(foreign_key="workflows.id", index=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    thread_id: str = Field(unique=True, index=True, max_length=255)
    title: Optional[str] = Field(default=None, max_length=255)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now, index=True)

    # Relationships
    workflow: Workflow = Relationship(back_populates="chat_sessions")
    user: User = Relationship(back_populates="chat_sessions")
    messages: list["WorkflowChatMessage"] = Relationship(back_populates="session")
    proposals: list["WorkflowProposal"] = Relationship(back_populates="session")

    @property
    def workflow_public_id(self) -> str:
        return make_workflow_public_id(self.workflow_id)


class WorkflowChatMessage(SQLModel, table=True):
    """Individual message in a chat session."""
    __tablename__ = "workflow_chat_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="workflow_chat_sessions.id", index=True)
    proposal_id: Optional[int] = Field(
        default=None,
        foreign_key="workflow_proposals.id",
        unique=True
    )
    role: str = Field(max_length=20)
    content: str = Field(sa_column=Column(Text))
    thinking: Optional[str] = Field(default=None, sa_column=Column(Text))
    suggested_edits: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    message_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSONB, name="metadata"))
    created_at: datetime = Field(default_factory=utc_now, index=True)

    # Relationships
    session: WorkflowChatSession = Relationship(back_populates="messages")


class WorkflowProposal(SQLModel, table=True):
    """Reviewable workflow edit proposal."""
    __tablename__ = "workflow_proposals"

    STATUS_PENDING: ClassVar[str] = "pending"
    STATUS_ACCEPTED: ClassVar[str] = "accepted"
    STATUS_REJECTED: ClassVar[str] = "rejected"

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_id: int = Field(foreign_key="workflows.id", index=True)
    session_id: Optional[int] = Field(
        default=None,
        foreign_key="workflow_chat_sessions.id",
        index=True
    )
    created_by_id: int = Field(foreign_key="users.id", index=True)
    summary: str = Field(max_length=512)
    spec: dict = Field(sa_column=Column(JSONB))
    status: str = Field(max_length=20, default=STATUS_PENDING)
    preview_graph: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    applied_graph: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    proposal_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSONB, name="metadata"))
    decided_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utc_now, index=True)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    workflow: Workflow = Relationship(back_populates="proposals")
    session: Optional[WorkflowChatSession] = Relationship(back_populates="proposals")
    created_by: User = Relationship(back_populates="workflow_proposals")

    @property
    def workflow_public_id(self) -> str:
        return make_workflow_public_id(self.workflow_id)


class TriggerSubscription(SQLModel, table=True):
    """Trigger configuration attached to a workflow."""
    __tablename__ = "trigger_subscriptions"
    __table_args__ = (
        Index('ix_trigger_sub_user_workflow', 'user_id', 'workflow_id'),
        Index('ix_trigger_sub_key_provider', 'trigger_key', 'provider_connection_id', 'enabled'),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    workflow_id: int = Field(foreign_key="workflows.id", index=True)
    trigger_key: str = Field(max_length=255)
    provider_connection_id: Optional[int] = None
    enabled: bool = Field(default=True)
    filters: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    bindings: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    provider_config: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    secret_token: Optional[str] = Field(default=None, max_length=255)
    poll_interval_seconds: int = Field(default=60)
    next_poll_at: datetime = Field(default_factory=utc_now)
    poll_cursor_json: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    poll_status: str = Field(default="ok", max_length=32)
    poll_error_json: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    poll_backoff_seconds: int = Field(default=0)
    poll_lock_owner: Optional[str] = Field(default=None, max_length=255)
    poll_lock_expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    user: User = Relationship(back_populates="trigger_subscriptions")
    workflow: Workflow = Relationship(back_populates="trigger_subscriptions")


class TriggerEvent(SQLModel, table=True):
    """Normalized incoming trigger event."""
    __tablename__ = "trigger_events"
    __table_args__ = (
        UniqueConstraint(
            'trigger_key', 'provider_connection_id', 'provider_event_id',
            name='uq_trigger_event_provider_id'
        ),
        UniqueConstraint(
            'trigger_key', 'provider_connection_id', 'event_hash',
            name='uq_trigger_event_hash'
        ),
        Index('ix_trigger_event_status', 'status', 'received_at'),
        Index('ix_trigger_event_provider', 'trigger_key', 'provider_connection_id'),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    trigger_key: str = Field(max_length=255)
    provider_connection_id: Optional[int] = None
    provider_event_id: Optional[str] = Field(default=None, max_length=255)
    event_hash: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Deterministic hash when provider_event_id unavailable"
    )
    occurred_at: Optional[datetime] = None
    received_at: datetime = Field(default_factory=utc_now)
    event: dict = Field(sa_column=Column(JSONB))
    raw_payload: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    status: TriggerEventStatus = Field(default=TriggerEventStatus.RECEIVED, max_length=20)
    error: Optional[dict] = Field(default=None, sa_column=Column(JSONB))


# ====================
# Integration Models
# ====================

class OAuthConnection(SQLModel, table=True):
    """Database model for storing OAuth connections/tokens."""
    __tablename__ = "oauth_connections"
    __table_args__ = (
        UniqueConstraint('user_id', 'provider', 'provider_account_id', name='uq_oauth_user_provider_account'),
    )

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True))
    user_id: int = Field(foreign_key="users.id")
    provider: str = Field(max_length=50)
    provider_account_id: str = Field(max_length=255)
    access_token_enc: str = Field(sa_column=Column(Text))
    refresh_token_enc: Optional[str] = Field(default=None, sa_column=Column(Text))
    expires_at: Optional[datetime] = None
    scopes: Optional[str] = Field(default=None, sa_column=Column(Text))
    token_type: str = Field(default="Bearer", max_length=50)
    provider_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    status: str = Field(default="active", max_length=20)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    user: User = Relationship(back_populates="oauth_connections")
    resources: list["IntegrationResource"] = Relationship(back_populates="oauth_connection")
    secrets: list["IntegrationSecret"] = Relationship(back_populates="oauth_connection")


class IntegrationResource(SQLModel, table=True):
    """Connected resources from integrations."""
    __tablename__ = "integration_resources"
    __table_args__ = (
        Index('ix_integration_resources_user_provider_type', 'user_id', 'provider', 'resource_type'),
        UniqueConstraint('oauth_connection_id', 'resource_type', 'resource_id', name='uq_resource_connection_type_id'),
    )

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True))
    user_id: int = Field(foreign_key="users.id", index=True)
    oauth_connection_id: Optional[int] = Field(default=None, foreign_key="oauth_connections.id")
    provider: str = Field(max_length=50)
    resource_type: str = Field(max_length=50)
    resource_id: str = Field(max_length=255)
    resource_key: Optional[str] = Field(default=None, max_length=255)
    name: Optional[str] = Field(default=None, max_length=255)
    resource_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    status: str = Field(default="active", max_length=20)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    user: User = Relationship(back_populates="integration_resources")
    oauth_connection: Optional[OAuthConnection] = Relationship(back_populates="resources")
    secrets: list["IntegrationSecret"] = Relationship(back_populates="resource")


class IntegrationSecret(SQLModel, table=True):
    """Generic vault for non-OAuth credentials."""
    __tablename__ = "integration_secrets"
    __table_args__ = (
        Index('ix_integration_secrets_user_provider_type', 'user_id', 'provider', 'secret_type'),
        UniqueConstraint('oauth_connection_id', 'name', name='uq_secret_connection_name'),
        UniqueConstraint('resource_id', 'name', name='uq_secret_resource_name'),
    )

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True))
    user_id: int = Field(foreign_key="users.id")
    provider: str = Field(max_length=50)
    oauth_connection_id: Optional[int] = Field(default=None, foreign_key="oauth_connections.id")
    resource_id: Optional[int] = Field(default=None, foreign_key="integration_resources.id")
    secret_type: str = Field(max_length=50)
    name: str = Field(max_length=100)
    value_enc: str = Field(sa_column=Column(Text))
    value_fingerprint: Optional[str] = Field(default=None, max_length=64)
    secret_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSONB, name="metadata"))
    expires_at: Optional[datetime] = None
    status: str = Field(default="active", max_length=20)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    user: User = Relationship(back_populates="integration_secrets")
    oauth_connection: Optional[OAuthConnection] = Relationship(back_populates="secrets")
    resource: Optional[IntegrationResource] = Relationship(back_populates="secrets")


# OLD WorkflowRecord model - keeping for backwards compat if needed
class WorkflowRecord(SQLModel, table=True):
    """Legacy workflow entity (pre-versioning system)."""
    __tablename__ = "workflow_records"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    name: str = Field(max_length=255)
    description: Optional[str] = Field(default=None, sa_column=Column(Text))
    spec: dict = Field(sa_column=Column(JSONB))
    version: int = Field(default=1)
    tags: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    meta: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    last_compile_ok: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now, index=True)

    # Relationships
    user: User = Relationship(back_populates="workflow_records")

    @property
    def workflow_id(self) -> str:
        return make_workflow_public_id(self.id)
