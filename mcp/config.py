"""MCP server configuration."""

from enum import Enum
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class MCPMode(str, Enum):
    """MCP server operating modes."""

    LOCAL = "local"  # Direct database access
    CLOUD = "cloud"  # API calls to cloud instance


class MCPConfig(BaseSettings):
    """MCP server configuration."""

    mode: MCPMode = Field(default=MCPMode.LOCAL, description="Operating mode (local or cloud)")
    api_url: str = Field(default="https://api.getseer.dev", description="Seer API URL for cloud mode")
    log_level: str = Field(default="info", description="Logging level")
    database_url: Optional[str] = Field(default=None, description="Database URL for local mode")

    class Config:
        env_prefix = "SEER_MCP_"
        case_sensitive = False

    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Load configuration from environment variables."""
        return cls()


def get_config() -> MCPConfig:
    """Get MCP configuration from environment."""
    return MCPConfig.from_env()
