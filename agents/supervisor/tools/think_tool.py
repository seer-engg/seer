from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agents.supervisor.tools.runtime_tool_store import _runtime_tool_store
from agents.supervisor.tools.integration_tools import get_available_integrations
from agents.supervisor.models import ToolDefinition
from pydantic import BaseModel, Field, create_model
import re
import logging
from typing import Dict, Optional, Any, Type, List

logger = logging.getLogger(__name__)

# LLM for extracting structured execution plan (lazy-loaded)
_extractor_llm = None

def _get_extractor_llm():
    """Lazy-load the extractor LLM."""
    global _extractor_llm
    if _extractor_llm is None:
        # Use config module which ensures OPENAI_API_KEY is available
        _extractor_llm = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0.0,
        )
    return _extractor_llm

from pydantic import Field, model_validator
from typing import Any

class ThinkInput(BaseModel):
    """Input model for think tool with case-insensitive field handling."""
    last_tool_call: str = Field(default="", description="What tool was just called and what it returned")
    scratchpad: str = Field(..., description="Your reasoning, plans, and analysis")
    
    @model_validator(mode='before')
    @classmethod
    def normalize_keys(cls, data: Any) -> Any:
        """Normalize field names to lowercase to handle case variations."""
        if isinstance(data, dict):
            normalized = {}
            for key, value in data.items():
                # Normalize key to lowercase
                normalized_key = key.lower()
                # Map common variations
                if normalized_key in ['scratchpad', 'scratch_pad']:
                    normalized['scratchpad'] = value
                elif normalized_key in ['last_tool_call', 'lasttoolcall', 'last_tool']:
                    normalized['last_tool_call'] = value
                else:
                    # Keep original key if not recognized
                    normalized[key] = value
            return normalized
        return data

@tool(args_schema=ThinkInput)
def think(
    last_tool_call: str,
    scratchpad: str,
) -> str:
    """Use this tool to think and reason about the current situation.
    
    **CRITICAL: You MUST call this tool BEFORE calling execute_tool.**
    Every execute_tool call MUST be preceded by a think() call that plans the execution.
    
    **REQUIRED: `last_tool_call` parameter**
    - You MUST provide `last_tool_call` on EVERY call
    - Format: "Tool: <tool_name>, Result: <what happened>"
    - For first call: "Tool: None, Result: Initial call"
    - After tool execution: "Tool: <tool_name>, Result: <success/error summary>"
    
    **BEFORE CALLING execute_tool - MANDATORY PLANNING:**
    When planning to call execute_tool, explicitly state in your scratchpad:
    1. Which tool you'll call (e.g., "I will call GITHUB_FIND_PULL_REQUESTS")
    2. ALL parameters you'll provide with reasoning for each:
       - "repo='seer-engg/buggy-coder' - because I need to search this specific repository"
       - "state='closed' - because I need closed/merged PRs"
       - "query='...' - because I need to search for..."
    3. Verify required params are provided with non-empty values
    
    **AFTER CALLING execute_tool - REFLECTION:**
    Reflect on results and plan next steps. Always include what tool was called and what happened.
    
    **FORMAT YOUR THINKING:**
    Use the scratchpad to:
    1. Analyze what just happened (last tool call and its result)
    2. Consider what this means for the current task/plan
    3. Decide what to do next (if planning execute_tool, state tool name and params with reasoning)
    
    This ensures you reason step-by-step rather than blindly executing tools."""
    
    # Extract planned execution if agent is planning execute_tool
    planned_execution = _extract_planned_execution(scratchpad)
    
    if planned_execution:
        # Store planned execution in runtime store (thread-scoped, using "default" for now)
        # TODO: Extract thread_id from runtime context if available
        _runtime_tool_store.store_planned_execution(planned_execution, thread_id="default")
        logger.debug("✅ Stored planned execution for tool: %s", planned_execution['tool_name'])
    
    # Return formatted response - always include last_tool_call
    parts = [f"Thought: {scratchpad}"]
    parts.append(f"Last tool: {last_tool_call}")
    return "\n".join(parts)


def _json_schema_type_to_python(json_type: str) -> Type:
    """Map JSON Schema type to Python type."""
    type_mapping = {
        'string': str,
        'integer': int,
        'number': float,
        'boolean': bool,
        'array': List[Any],  # Could be more specific if items type available
        'object': Dict[str, Any],
    }
    return type_mapping.get(json_type.lower(), str)  # Default to str


def _create_tool_params_model(tool_schema: ToolDefinition) -> Type[BaseModel]:
    """Dynamically create Pydantic model from tool schema.
    
    Args:
        tool_schema: ToolDefinition with parameters
        
    Returns:
        Dynamic Pydantic model class for tool parameters
        
    Raises:
        ValueError: If tool has no schema (should not happen - schema required)
    """
    from typing import Optional
    
    field_definitions = {}
    
    for param in tool_schema.parameters:
        # Map JSON Schema types to Python types
        python_type = _json_schema_type_to_python(param.type)
        
        if param.required:
            field_definitions[param.name] = (
                python_type, 
                Field(description=param.description)
            )
        else:
            field_definitions[param.name] = (
                Optional[python_type], 
                Field(default=None, description=param.description)
            )
    
    # Create model name from tool name (sanitize for Python class name)
    model_name = f"{tool_schema.name.replace('.', '_').replace('-', '_')}Params"
    
    if not field_definitions:
        # Empty model for tools with no parameters
        return create_model(model_name)
    
    return create_model(model_name, **field_definitions)


def _create_execution_plan_model(tool_schema: ToolDefinition) -> Type[BaseModel]:
    """Create execution plan model with tool-specific params model.
    
    Args:
        tool_schema: ToolDefinition with parameters
        
    Returns:
        Dynamic Pydantic model class with tool_name, reasoning, and params
    """
    params_model = _create_tool_params_model(tool_schema)
    
    model_name = f"{tool_schema.name.replace('.', '_').replace('-', '_')}ExecutionPlan"
    
    return create_model(
        model_name,
        tool_name=(str, Field(description="Tool name")),
        reasoning=(str, Field(description="Why this tool is needed")),
        params=(params_model, Field(description="Tool parameters"))
    )


def _extract_planned_execution(scratchpad: str) -> Optional[Dict[str, Any]]:
    """
    Extract planned tool execution using dynamic Pydantic model from tool schema.
    
    **SCHEMA REQUIRED:** This function requires a tool schema. If no schema is found,
    the tool cannot be executed and None is returned.
    
    Returns:
        Dict with keys: tool_name, reasoning, params (dict)
        None if no tool detected or schema not found
    """
    # **STEP 1: Try to detect tool name (lightweight - regex)**
    # Skip built-in tools that don't require planning/schema validation
    if "search_tools" in scratchpad.lower() or "write_todos" in scratchpad.lower() or "spawn_worker" in scratchpad.lower():
        return None

    integrations = get_available_integrations()
    integration_patterns = [f"{i.upper()}_\\w+" for i in integrations]
    pattern = f"({'|'.join(integration_patterns)})"
    tool_name_match = re.search(pattern, scratchpad, re.IGNORECASE)
    if not tool_name_match and "execute_tool" not in scratchpad.lower():
        return None
    
    # **STEP 2: Get tool schema - REQUIRED (no fallback)**
    tool_name = tool_name_match.group(1) if tool_name_match else None
    if not tool_name:
        # Try to extract from scratchpad text if regex didn't match
        # Look for patterns like "I will call TOOL_NAME" or "call TOOL_NAME"
        tool_name_match = re.search(r'(?:call|execute|use)\s+([A-Z_][A-Z0-9_]+)', scratchpad, re.IGNORECASE)
        if tool_name_match:
            tool_name = tool_name_match.group(1)
    
    if not tool_name:
        logger.debug("No tool name detected in scratchpad")
        return None
    
    tool_schema = _runtime_tool_store.get_tool_schema(tool_name)
    
    # **SCHEMA REQUIRED - NO FALLBACK**
    if not tool_schema:
        logger.warning("❌ Tool '%s' has no schema. Cannot execute. Agent must use a different tool.", tool_name)
        return None
    
    # **STEP 3: Create dynamic Pydantic model from schema**
    try:
        execution_plan_model = _create_execution_plan_model(tool_schema)
    except Exception as e:
        logger.error(f"Failed to create dynamic model for {tool_name}: {e}")
        return None
    
    # **STEP 4: Build prompt with schema context**
    required_params = [p for p in tool_schema.parameters if p.required]
    optional_params = [p for p in tool_schema.parameters if not p.required]
    
    schema_section = f"""
**TOOL SCHEMA:**
Tool: {tool_schema.name}
Description: {tool_schema.description}

**REQUIRED Parameters (you MUST provide all of these):**
{chr(10).join(f"- {p.name} ({p.type}): {p.description}" for p in required_params) if required_params else "None"}

**Optional Parameters:**
{chr(10).join(f"- {p.name} ({p.type}): {p.description}" for p in optional_params) if optional_params else "None"}
"""
    
    
    # **STEP 5: Extract using dynamic Pydantic model**
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Extract tool execution plan from the agent's thinking.

{schema_section}

**CRITICAL INSTRUCTIONS:**
1. Extract the exact tool name: "{tool_schema.name}"
2. Extract ALL parameters the agent mentions or plans to use
3. **You MUST provide ALL required parameters with non-empty values**
4. Use proper types: strings as strings, numbers as numbers, booleans as booleans

The tool schema above defines the exact structure. Follow it precisely."""),
        ("human", "{scratchpad}")
    ])
    
    try:
        extractor_llm = _get_extractor_llm()
        chain = extraction_prompt | extractor_llm.with_structured_output(
            execution_plan_model,
            method="function_calling"
        )
        result = chain.invoke({"scratchpad": scratchpad})
        
        # Extract params dict from Pydantic model
        if hasattr(result, 'params'):
            params_dict = result.params.model_dump() if hasattr(result.params, 'model_dump') else dict(result.params)
        else:
            params_dict = {}
        
        # Return dict format for storage
        return {
            "tool_name": result.tool_name,
            "reasoning": result.reasoning,
            "params": params_dict
        }
        
    except Exception as e:
        logger.error("Failed to extract planned execution for %s: %s", tool_name, e)
        import traceback
        logger.debug(traceback.format_exc())
        return None

