from pydantic import BaseModel, Field
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Pydantic models for structured outputs
class Verdict(BaseModel):
    """Evaluator's verdict on the Actor's response"""
    passed: bool = Field(description="Whether the response passes evaluation")
    score: float = Field(ge=0.0, le=1.0, description="Quality score from 0.0 to 1.0")
    reasoning: str = Field(description="Detailed explanation of the judgment")
    issues: list[str] = Field(default_factory=list, description="Specific problems found")
    execution_results: str = Field(default="", description="Actual execution results from sandbox")


class Reflection(BaseModel):
    """Reflection agent's feedback for improvement"""
    key_issues: list[str] = Field(description="Main problems identified")
    suggestions: list[str] = Field(description="Specific, actionable improvements")
    focus_areas: list[str] = Field(description="What to prioritize in next attempt")
    examples: list[str] = Field(default_factory=list, description="Concrete examples if helpful")


class ReflexionState(TypedDict, total=False):
    """State for the reflexion agent graph"""
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Current attempt tracking
    current_attempt: int
    max_attempts: int
    
    # Evaluator's verdict
    evaluator_verdict: Verdict | None
    
    # Final result status
    success: bool
    
    # Memory store key for this conversation (e.g., user_id or domain)
    memory_key: str

