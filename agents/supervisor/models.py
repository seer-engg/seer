"""
Pydantic models for Supervisor.
Replaces string checks, dict access, and regex parsing with type-safe models.
"""
from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

# ============================================================================
# Worker Response Models
# ============================================================================

class WorkerStatus(str, Enum):
    """Status of a worker execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"

class WorkerResponse(BaseModel):
    """Structured response from a worker agent."""
    status: WorkerStatus
    message: str
    error: Optional[str] = Field(default=None, description="Error message if status is failure")
    tool_calls_made: int = Field(default=0, description="Number of tool calls executed")
    
    @classmethod
    def from_message_content(cls, content: str, messages: List[BaseMessage] = None) -> "WorkerResponse":
        """Parse a worker's final message into a structured response."""
        # Count tool calls
        tool_calls = 0
        if messages:
            tool_calls = sum(1 for m in messages if hasattr(m, 'tool_calls') and m.tool_calls)
        
        # Check for explicit status indicators
        content_lower = content.lower()
        if "❌" in content or "error" in content_lower or "failed" in content_lower:
            return cls(
                status=WorkerStatus.FAILURE,
                message=content,
                error=content,
                tool_calls_made=tool_calls
            )
        elif "✅" in content or "success" in content_lower or "completed" in content_lower:
            return cls(
                status=WorkerStatus.SUCCESS,
                message=content,
                tool_calls_made=tool_calls
            )
        else:
            # Default to partial if unclear
            return cls(
                status=WorkerStatus.PARTIAL,
                message=content,
                tool_calls_made=tool_calls
            )

# ============================================================================
# Evaluation Models
# ============================================================================

class EvaluationResult(BaseModel):
    """Result of task evaluation."""
    success: bool
    reasoning: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in evaluation (0.0-1.0)")

# ============================================================================
# Benchmark Models
# ============================================================================

class BenchmarkResult(BaseModel):
    """Result of a benchmark run."""
    condition: Literal["baseline", "baseline_delegate"]
    success: bool
    execution_time: float = Field(ge=0, description="Execution time in seconds")
    total_tokens_estimate: int = Field(default=0, ge=0)
    context_size_estimate: int = Field(default=0, ge=0, description="Peak context size in characters")
    tool_calls: int = Field(default=0, ge=0)
    reasoning: str
    trace_id: str

# ============================================================================
# Tool Parameter Models (Already defined but ensuring consistency)
# ============================================================================

class ToolParameter(BaseModel):
    """Parameter definition for a tool."""
    name: str
    type: str
    description: str = ""
    required: bool = False

class ToolDefinition(BaseModel):
    """Definition of a tool from search results."""
    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
