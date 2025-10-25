from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.messages import AnyMessage
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
    coding_context: str = Field(description="Context of the coding task that failed the evaluation and this reflection is applicable for", example="Python code for merging intervals")
    reflection: str = Field(description="Key reflection points to be considered to avoid the issues in the future")



class IOState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]

    # Current attempt tracking


class ReflexionState(IOState):
    trajectory: list[AnyMessage] = []
    current_attempt: int = Field(default=0, description="Current attempt number")
    max_attempts: int = Field(default=2, description="Maximum number of attempts to act with the environment")
    
    # Evaluator's verdict
    evaluator_verdict: Verdict = Field(default=Verdict(passed=False, score=0.0, reasoning="", issues=[]), description="Evaluator's verdict on the Actor's response")
    
    # Final result status
    success: bool = Field(default=False, description="Whether the response passed evaluation")
    
    # Memory store key for this conversation (e.g., user_id or domain)
    memory_key: str = Field(default="user_1234567890", description="Memory store key for this conversation")
    pass
