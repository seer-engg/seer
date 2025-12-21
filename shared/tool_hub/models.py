from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ToolFunction(BaseModel):
    """Represents the function part of an OpenAI tool definition."""
    name: str
    description: Optional[str] = ""
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class Tool(BaseModel):
    """Represents a tool input (compatible with OpenAI tool schema)."""
    type: str = "function"
    function: ToolFunction
    executable: Optional[Any] = Field(default=None, exclude=True, description="The executable callable/tool object")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        """Helper to create a Tool from a dictionary."""
        # Handle cases where the dict is just the function part or the full tool part
        if "function" in data:
            return cls(**data)
        else:
            # Assume it's the function definition directly (common in some frameworks)
            return cls(function=ToolFunction(**data))
    
    class Config:
        arbitrary_types_allowed = True

class EnrichedTool(BaseModel):
    """Internal representation with enriched metadata."""
    name: str
    description: str
    parameters: Dict[str, Any]
    use_cases: List[str] = Field(description="Specific user intent questions this tool solves")
    likely_neighbors: List[str] = Field(description="Tools likely to be used immediately before or after")
    required_params: List[str] = Field(description="Parameter names required to use this tool")
    embedding_text: str = Field(description="Text used for vector embedding")
    original_tool: Tool = Field(description="The original tool object")
    
    def get_executable(self) -> Optional[Any]:
        return self.original_tool.executable
