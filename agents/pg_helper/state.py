"""State definition for the Supervisor agent."""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class FileAttachment(BaseModel):
    """Represents a file attachment (image, PDF, etc.)."""
    
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type (e.g., 'image/png', 'application/pdf')")
    data: str = Field(..., description="Base64-encoded file content")
    
    @property
    def is_image(self) -> bool:
        """Check if the file is an image."""
        return self.content_type.startswith("image/")
    
    @property
    def is_pdf(self) -> bool:
        """Check if the file is a PDF."""
        return self.content_type == "application/pdf"


class SupervisorInput(BaseModel):
    """Input schema for the Supervisor agent."""
    
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="Conversation messages including multimodal content"
    )
    files: List[FileAttachment] = Field(
        default_factory=list,
        description="Attached files (images, PDFs) for processing"
    )
    database_connection_string: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string. Uses config.database_uri if not provided."
    )


class SupervisorOutput(BaseModel):
    """Output schema for the Supervisor agent."""
    
    response: Optional[str] = Field(
        default=None,
        description="Final response from the supervisor"
    )
    query_results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Results from database queries if any"
    )


class SupervisorState(SupervisorInput, SupervisorOutput):
    """
    State for the Supervisor agent.
    
    Supports:
    - Multimodal messages (text, images, PDFs)
    - Database operations via postgres tools
    - Subagent orchestration
    - Human-in-the-loop interrupts
    """
    
    # Internal state for tracking
    pending_file_request: Optional[str] = Field(
        default=None,
        description="Description of files being requested via interrupt"
    )
    
    # Database context
    current_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cached database schema information"
    )

