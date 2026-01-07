# 🔥 Clean Slate Migration: Tortoise → SQLModel (YOLO Edition)

**Status:** READY TO EXECUTE
**Timeline:** 1-2 days (aggressive, clean break)
**Risk Level:** ZERO (no users = no data to lose)
**Approach:** Nuke everything, start fresh

---

## 🎯 Strategy: Clean Break

Since you have:
- ✅ 0 production users
- ✅ Internal usage only
- ✅ Railway dev + main environments
- ✅ Can test on dev first

**We're going FULL SEND:**
1. ❌ Remove Tortoise + Aerich completely
2. ✅ Set up SQLModel + Alembic fresh
3. ✅ Convert ALL models at once
4. ✅ Test locally with fresh DB
5. ✅ Deploy to Railway dev → test → main

**No incremental bullshit. Clean. Simple. Fast.**

---

## Phase 1: Rip Out Tortoise (30 min)

### 1.1 Remove Dependencies

```bash
cd /Users/pika/Projects/seer

# Uninstall old crap
uv remove tortoise-orm aerich

# Add new hotness
uv add sqlmodel alembic asyncpg
```

**Update `pyproject.toml`:**
```toml
dependencies = [
    # ... keep everything except these:
    # ❌ "tortoise-orm[asyncpg]>=0.25.2",  # DELETED
    # ❌ "aerich>=0.9.2",  # DELETED

    # ✅ New stack:
    "sqlmodel>=0.0.22",
    "alembic>=1.14.0",
    "asyncpg>=0.31.0",  # Keep for async Postgres

    # ... rest of deps stay the same
]

# ❌ Delete this section:
# [tool.aerich]
# tortoise_orm = "shared.database.config.TORTOISE_ORM"
# location = "./migrations"
# src_folder = "./."
```

### 1.2 Delete Old Files

```bash
# Nuke Aerich migrations
rm -rf migrations/

# Delete Tortoise config
rm shared/database/config.py

# We'll replace workflow_models.py with sqlmodel_models.py
# (keep it for reference during conversion, delete after)
```

---

## Phase 2: Set Up SQLModel + Alembic (30 min)

### 2.1 Initialize Alembic

```bash
# From project root
alembic init alembic

# This creates:
# - alembic.ini (config)
# - alembic/ (migrations folder)
```

### 2.2 Configure Alembic

**Edit `alembic.ini`:**
```ini
# Line ~58: Comment out hardcoded URL (we'll use env var)
# sqlalchemy.url = driver://user:pass@localhost/dbname

# Line ~63: Better file naming
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(slug)s

# Line ~76: Enable autogenerate features
# (uncomment if commented)
```

**Edit `alembic/env.py`:**
```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import SQLModel base and all models
from shared.database.base import Base
from shared.database import models  # This will import all SQLModel models

# Alembic Config object
config = context.config

# Get DATABASE_URL from environment
from shared.config import config as app_config
database_url = app_config.DATABASE_URL

# Convert postgresql:// to postgresql+asyncpg:// for SQLAlchemy
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

config.set_main_option("sqlalchemy.url", database_url)

# Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### 2.3 Create Database Infrastructure

**Create `shared/database/base.py`:**
```python
"""SQLModel database configuration."""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlmodel import SQLModel
from shared.config import config

# Create async engine
engine = create_async_engine(
    config.DATABASE_URL,
    echo=False,  # Set True for SQL debugging
    future=True,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=10,
)

# Create session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for all models
Base = SQLModel


async def get_session():
    """FastAPI dependency for database sessions."""
    async with async_session_maker() as session:
        yield session


async def init_db():
    """Create all tables (for development only)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Dispose of engine connections."""
    await engine.dispose()
```

---

## Phase 3: Convert Models to SQLModel (2-3 hours)

Now the fun part - converting your Tortoise models to SQLModel!

### 3.1 Create SQLModel Models

**Create `shared/database/models.py`:**
```python
"""
SQLModel models - clean slate conversion from Tortoise.

All models use SQLModel which integrates with:
- Pydantic (validation)
- SQLAlchemy (database)
- FastAPI (automatic API schemas)
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel, Relationship, Column
from sqlalchemy import Index, UniqueConstraint, String, Integer
from sqlalchemy.dialects.postgresql import JSONB


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
    """User model."""
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True, max_length=255)
    name: Optional[str] = Field(default=None, max_length=255)
    avatar_url: Optional[str] = Field(default=None, max_length=500)
    clerk_user_id: Optional[str] = Field(default=None, unique=True, max_length=255)
    signup_source: Optional[str] = Field(default=None, max_length=50)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    workflows: list["Workflow"] = Relationship(back_populates="user")
    workflow_runs: list["WorkflowRun"] = Relationship(back_populates="user")


class Workflow(SQLModel, table=True):
    """Workflow entity - owns drafts, versions, and published state."""
    __tablename__ = "workflows"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    name: str = Field(max_length=255)
    description: Optional[str] = None
    tags: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    meta: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    published_version_id: Optional[int] = Field(
        default=None,
        foreign_key="workflow_versions.id"
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Relationships
    user: User = Relationship(back_populates="workflows")
    versions: list["WorkflowVersion"] = Relationship(back_populates="workflow")
    draft: Optional["WorkflowDraft"] = Relationship(back_populates="workflow")
    runs: list["WorkflowRun"] = Relationship(back_populates="workflow")

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
    status: WorkflowVersionStatus = Field(default=WorkflowVersionStatus.DRAFT)
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
    source: WorkflowRunSource = Field(default=WorkflowRunSource.MANUAL)
    subscription_id: Optional[int] = Field(
        default=None,
        foreign_key="trigger_subscriptions.id"
    )
    trigger_event_id: Optional[int] = Field(
        default=None,
        foreign_key="trigger_events.id"
    )
    status: WorkflowRunStatus = Field(default=WorkflowRunStatus.QUEUED)
    output: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    error: Optional[str] = None
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
    content: str
    thinking: Optional[str] = None
    suggested_edits: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    metadata: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=utc_now, index=True)


class WorkflowProposal(SQLModel, table=True):
    """Reviewable workflow edit proposal."""
    __tablename__ = "workflow_proposals"

    STATUS_PENDING = "pending"
    STATUS_ACCEPTED = "accepted"
    STATUS_REJECTED = "rejected"

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
    metadata: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    decided_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utc_now, index=True)
    updated_at: datetime = Field(default_factory=utc_now)

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
    status: TriggerEventStatus = Field(default=TriggerEventStatus.RECEIVED)
    error: Optional[dict] = Field(default=None, sa_column=Column(JSONB))


# ====================
# Integration Models
# ====================

class IntegrationSecret(SQLModel, table=True):
    """OAuth tokens and API keys for integrations."""
    __tablename__ = "integration_secrets"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    provider: str = Field(max_length=50, index=True)
    encrypted_data: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expires_at: Optional[datetime] = None


class IntegrationResource(SQLModel, table=True):
    """Connected resources from integrations."""
    __tablename__ = "integration_resources"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    provider: str = Field(max_length=50, index=True)
    resource_type: str = Field(max_length=100)
    resource_id: str = Field(max_length=255)
    resource_data: dict = Field(sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


# OLD WorkflowRecord model - keeping for backwards compat if needed
class WorkflowRecord(SQLModel, table=True):
    """Legacy workflow entity (pre-versioning system)."""
    __tablename__ = "workflow_records"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    name: str = Field(max_length=255)
    description: Optional[str] = None
    spec: dict = Field(sa_column=Column(JSONB))
    version: int = Field(default=1)
    tags: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    meta: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    last_compile_ok: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now, index=True)

    @property
    def workflow_id(self) -> str:
        return make_workflow_public_id(self.id)
```

### 3.2 Update Database Init

**Replace `shared/database/__init__.py`:**
```python
"""Database initialization with SQLModel."""
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI
import subprocess

from shared.logger import get_logger
from shared.database.base import init_db, close_db
from shared.config import config

# Import all models to ensure they're registered
from shared.database import models  # noqa

logger = get_logger("shared.database")


async def run_alembic_migrations() -> None:
    """Run Alembic migrations."""
    if not config.AUTO_APPLY_DATABASE_MIGRATIONS:
        logger.info("⏭️  Auto-migrations disabled. Skipping Alembic.")
        return

    try:
        logger.info("🔄 Running Alembic migrations...")
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            check=True,
            cwd="/Users/pika/Projects/seer"
        )
        logger.info("✅ Alembic migrations completed successfully")
        if result.stdout:
            logger.debug(f"Alembic output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Alembic migration failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"❌ Alembic error: {e}")
        raise


@asynccontextmanager
async def db_lifespan(_: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler for database."""
    logger.info("🚀 Initializing database...")

    # Run migrations
    await run_alembic_migrations()

    # Initialize connection pool
    # (tables are created by migrations, not by init_db)
    # await init_db()  # Only use in development if you want auto-create

    logger.info("✅ Database ready")

    try:
        yield
    finally:
        logger.info("🔌 Closing database connections...")
        await close_db()
        logger.info("✅ Database closed")


__all__ = [
    "db_lifespan",
    "models",
]
```

---

## Phase 4: Update API Code (1-2 hours)

### 4.1 Update All Imports

**Find and replace:**
```bash
# Search for Tortoise imports
grep -r "from shared.database.workflow_models import" . --include="*.py"
grep -r "from tortoise" . --include="*.py"

# Replace with:
# from shared.database.models import ...
# from sqlmodel import select
# from shared.database.base import get_session
```

### 4.2 Update Query Patterns

**Old Tortoise style:**
```python
# ❌ OLD
from shared.database.workflow_models import Workflow

workflows = await Workflow.filter(user_id=user.id).all()
workflow = await Workflow.get(id=workflow_id)
```

**New SQLModel style:**
```python
# ✅ NEW
from sqlmodel import select
from shared.database.models import Workflow
from shared.database.base import get_session

# In FastAPI route:
@router.get("/workflows")
async def get_workflows(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    statement = select(Workflow).where(Workflow.user_id == user.id)
    result = await session.execute(statement)
    workflows = result.scalars().all()
    return workflows

# Get by ID:
statement = select(Workflow).where(Workflow.id == workflow_id)
result = await session.execute(statement)
workflow = result.scalar_one_or_none()
```

### 4.3 Update FastAPI Routes

**Example conversion - workflow routes:**

```python
# api/routes/workflows.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from shared.database.base import get_session
from shared.database.models import Workflow, WorkflowVersion, User

router = APIRouter(prefix="/workflows", tags=["Workflows"])


@router.get("/")
async def list_workflows(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """List user's workflows."""
    statement = select(Workflow).where(Workflow.user_id == user.id)
    result = await session.execute(statement)
    workflows = result.scalars().all()
    return workflows


@router.post("/", response_model=Workflow)
async def create_workflow(
    name: str,
    description: str | None = None,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """Create new workflow."""
    workflow = Workflow(
        user_id=user.id,
        name=name,
        description=description
    )
    session.add(workflow)
    await session.commit()
    await session.refresh(workflow)
    return workflow


@router.get("/{workflow_id}")
async def get_workflow(
    workflow_id: int,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """Get workflow by ID."""
    statement = (
        select(Workflow)
        .where(Workflow.id == workflow_id)
        .where(Workflow.user_id == user.id)
    )
    result = await session.execute(statement)
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return workflow


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: int,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """Delete workflow."""
    statement = (
        select(Workflow)
        .where(Workflow.id == workflow_id)
        .where(Workflow.user_id == user.id)
    )
    result = await session.execute(statement)
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    await session.delete(workflow)
    await session.commit()
    return {"message": "Workflow deleted"}
```

**Do this for ALL routes that use database.**

---

## Phase 5: Generate Initial Migration (15 min)

### 5.1 Create First Migration

```bash
# Generate migration from SQLModel models
alembic revision --autogenerate -m "initial_schema"

# This creates: alembic/versions/20260106_XXXX_initial_schema.py
```

### 5.2 Review Migration

```bash
# Open the generated file
cat alembic/versions/*_initial_schema.py

# Should see CREATE TABLE statements for ALL your models
# Review to make sure it looks correct
```

---

## Phase 6: Test Locally (30 min)

### 6.1 Nuke Local Database

```bash
# Stop containers
docker-compose down

# Delete postgres volume (nukes all data)
docker volume rm seer_postgres_data

# Start fresh
docker-compose up -d postgres redis

# Wait for postgres
sleep 5
```

### 6.2 Run Migrations

```bash
# Apply migrations
alembic upgrade head

# Check tables
docker exec -it seer-postgres psql -U postgres -d seer -c "\dt"

# Should see ALL your tables created fresh!
```

### 6.3 Start Application

```bash
docker-compose up langgraph-server

# Watch logs for:
# ✅ Alembic migrations completed
# ✅ Database ready
# No Tortoise/Aerich errors!
```

### 6.4 Test API Endpoints

```bash
# Test user creation, workflow CRUD, etc.
curl -X POST http://localhost:8000/workflows/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Workflow", "description": "Fresh SQLModel test"}'

# Should work!
```

---

## Phase 7: Deploy to Railway Dev (30 min)

### 7.1 Prepare Railway Deployment

**Update `railway.toml` (if you have one):**
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uv run alembic upgrade head && uv run uvicorn api.main:app --host 0.0.0.0 --port $PORT"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

**Or update your Dockerfile CMD:**
```dockerfile
CMD ["sh", "-c", "alembic upgrade head && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
```

### 7.2 Push to Dev Branch

```bash
git checkout dev  # Or your dev branch
git add .
git commit -m "BREAKING: Migrate from Tortoise to SQLModel + Alembic

- Remove Tortoise ORM and Aerich
- Add SQLModel + Alembic for migrations
- Convert all models to SQLModel
- Clean slate - requires fresh database

⚠️ DATABASE WILL BE WIPED ON NEXT DEPLOY
"
git push origin dev
```

### 7.3 Wipe Railway Dev Database

**In Railway Dashboard:**
1. Go to `seer-dev` project
2. Find PostgreSQL service
3. Click "Variables" → Find `DATABASE_URL`
4. Click "..." → "Reset Database" or just delete the volume

**Or via CLI:**
```bash
# If you have railway CLI
railway login
railway link  # Link to dev project
railway run psql -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
```

### 7.4 Trigger Deploy

Railway will auto-deploy on push. Watch logs:
```
🔄 Running Alembic migrations...
✅ Alembic migrations completed
✅ Database ready
```

### 7.5 Test Dev Environment

```bash
# Get dev URL from Railway dashboard
curl https://your-dev-url.railway.app/health

# Test workflows endpoint
curl https://your-dev-url.railway.app/workflows/
```

**SUCCESS = SQLModel works on Railway!** 🎉

---

## Phase 8: Deploy to Production (30 min)

Once dev is stable for a day or two:

### 8.1 Merge to Main

```bash
git checkout main
git merge dev
git push origin main
```

### 8.2 Wipe Production Database

**⚠️ LAST CHANCE TO BACK UP** (but you said YOLO)

In Railway dashboard:
1. Go to `seer-main` project
2. PostgreSQL service → Reset database

### 8.3 Deploy

Railway auto-deploys main branch.

### 8.4 Verify Production

```bash
curl https://your-prod-url.railway.app/health
curl https://your-prod-url.railway.app/workflows/
```

**DONE! 🚀**

---

## 🎉 Success Checklist

- [ ] Tortoise + Aerich removed from dependencies
- [ ] SQLModel + Alembic installed
- [ ] All models converted to SQLModel
- [ ] All API routes updated to use SQLModel queries
- [ ] Initial Alembic migration created
- [ ] Tested locally with fresh DB
- [ ] Deployed to Railway dev - working
- [ ] Deployed to Railway main - working
- [ ] No more MODELS_STATE conflicts ever again!

---

## 🆘 If Something Breaks

### Locally:
```bash
# Nuke and restart
docker-compose down
docker volume rm seer_postgres_data
docker-compose up
```

### Railway:
```bash
# Redeploy with fresh DB
# (Railway keeps your code in git, just reset DB)
```

### Emergency Rollback:
```bash
# Revert git commit
git revert HEAD
git push

# Old Tortoise code comes back
# (But you'll need to reinstall dependencies)
```

---

## 📊 Timeline

| Phase | Time | What |
|-------|------|------|
| 1. Remove Tortoise | 30 min | Delete deps, files |
| 2. Setup SQLModel | 30 min | Install, configure |
| 3. Convert Models | 2-3 hours | Big model file |
| 4. Update API Code | 1-2 hours | Route conversions |
| 5. Generate Migration | 15 min | Alembic autogen |
| 6. Test Locally | 30 min | Fresh DB test |
| 7. Deploy Dev | 30 min | Railway dev |
| 8. Deploy Prod | 30 min | Railway main |

**Total: 6-8 hours active work** (split over 1-2 days)

---

## 🚀 READY TO GO?

This is the clean break approach. No complexity. Just:
1. Rip out old
2. Put in new
3. Test
4. Deploy

Want me to start **Phase 1** right now?
