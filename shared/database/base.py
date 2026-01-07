"""SQLModel database configuration."""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlmodel import SQLModel
from shared.config import config

# Get DATABASE_URL or use default for local dev
database_url = config.DATABASE_URL or "postgresql+asyncpg://postgres:postgres@localhost:5432/seer"

# Create async engine
engine = create_async_engine(
    database_url,
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

# Base class for all models (SQLModel inherits from both SQLAlchemy Base and Pydantic BaseModel)
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
