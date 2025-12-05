from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tools.runtime_tool_store import _runtime_tool_store
from tools.secrets_store import _secrets_store
from models import ToolExecutionPlanBase
import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# LLM for extracting structured execution plan (lazy-loaded)
_extractor_llm = None

def _get_extractor_llm():
    """Lazy-load the extractor LLM."""
    global _extractor_llm
    if _extractor_llm is None:
        _extractor_llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
    return _extractor_llm

def _get_all_env_vars() -> Dict[str, str]:
    """
    Get secrets from secrets store (only specific vars from .env).
    Returns dict of {var_name: value} with actual values.
    """
    return _secrets_store.get_all()

@tool
def think(
    scratchpad: str,
    last_tool_call: str = "",  # Optional: What tool was just called and what it returned
) -> str:
    """Use this tool to think and reason about the current situation.
    
    **CRITICAL: You MUST call this tool BEFORE calling execute_tool.**
    Every execute_tool call MUST be preceded by a think() call that plans the execution.
    
    **BEFORE CALLING execute_tool - MANDATORY PLANNING:**
    When planning to call execute_tool, explicitly state in your scratchpad:
    1. Which tool you'll call (e.g., "I will call GITHUB_FIND_PULL_REQUESTS")
    2. ALL parameters you'll provide with reasoning for each:
       - "repo='seer-engg/buggy-coder' - because I need to search this specific repository"
       - "state='closed' - because I need closed/merged PRs"
       - "query='...' - because I need to search for..."
    3. Verify required params are provided with non-empty values
    
    **AFTER CALLING execute_tool - REFLECTION:**
    Reflect on results and plan next steps.
    
    **FORMAT YOUR THINKING:**
    Use the scratchpad to:
    1. Analyze what just happened (last tool call and its result)
    2. Consider what this means for the current task/plan
    3. Decide what to do next (if planning execute_tool, state tool name and params with reasoning)
    
    This ensures you reason step-by-step rather than blindly executing tools."""
    
    # Extract planned execution if agent is planning execute_tool
    planned_execution = _extract_planned_execution(scratchpad)
    
    if planned_execution:
        # Store planned execution in runtime store
        _runtime_tool_store.store_planned_execution(planned_execution)
        logger.debug(f"✅ Stored planned execution for tool: {planned_execution.tool_name}")
    
    # Return formatted response
    parts = [f"Thought: {scratchpad}"]
    if last_tool_call:
        parts.append(f"Last tool: {last_tool_call}")
    return "\n".join(parts)


def _extract_planned_execution(scratchpad: str) -> ToolExecutionPlanBase | None:
    """
    Use LLM with structured output to extract planned tool execution from scratchpad.
    Single LLM call with tool schema context for better validation.
    """
    # **STEP 1: Try to detect tool name (lightweight - regex)**
    tool_name_match = re.search(r'(GITHUB_\w+|ASANA_\w+)', scratchpad)
    if not tool_name_match and "execute_tool" not in scratchpad.lower():
        return None
    
    # If we found a tool name, get its schema
    tool_name = tool_name_match.group(1) if tool_name_match else None
    tool_schema = None
    if tool_name:
        tool_schema = _runtime_tool_store.get_tool_schema(tool_name)
    
    # Build schema section for prompt
    schema_section = ""
    if tool_schema:
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
    else:
        schema_section = """
**TOOL SCHEMA:**
Tool name not detected or schema not found. Extract tool name from scratchpad if mentioned.
"""
    
    # Get all environment variables
    env_vars = _get_all_env_vars()
    env_vars_section = ""
    if env_vars:
        # Escape curly braces in env var values to prevent LangChain template parsing
        def escape_braces(value: str) -> str:
            """Escape curly braces for LangChain template."""
            return str(value).replace("{", "{{").replace("}", "}}")
        
        env_vars_list = "\n".join(
            f"- {key}: {escape_braces(value)}" 
            for key, value in sorted(env_vars.items())
        )
        env_vars_section = f"""

**AVAILABLE ENVIRONMENT VARIABLES:**
The following environment variables are available and can be used for tool parameters:
{env_vars_list}

**IMPORTANT:** If a required parameter matches an env var name or pattern (e.g., `workspace_gid` → check for `ASANA_WORKSPACE_GID` or similar), 
use the env var value directly. Check env vars first before using placeholders or asking the user.
"""
    
    # **SINGLE LLM CALL: Extract tool name and params with schema context**
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Extract tool execution plan from the agent's thinking.

{schema_section}
{env_vars_section}

**CRITICAL INSTRUCTIONS:**
1. Extract the exact tool name the agent is planning to call (e.g., "GITHUB_FIND_PULL_REQUESTS")
2. Extract ALL parameters the agent mentions or plans to use
3. **If tool schema is provided above:**
   - You MUST provide ALL required parameters with non-empty values
   - Check env vars first for parameter values
   - If no env var exists, use placeholder like WORKSPACE_GID_PLACEHOLDER with reasoning
4. For each parameter, provide reasoning explaining why that value is chosen

If NOT planning a tool call, set tool_name to empty string."""),
        ("human", "{scratchpad}")
    ])
    
    try:
        extractor_llm = _get_extractor_llm()
        chain = extraction_prompt | extractor_llm.with_structured_output(
            ToolExecutionPlanBase,
            method="function_calling"
        )
        result = chain.invoke({"scratchpad": scratchpad})
        
        # If no tool call detected, return None
        if not result.tool_name:
            return None
        
        # **VALIDATE AGAINST SCHEMA** (double-check, but LLM should have gotten it right)
        tool_schema = _runtime_tool_store.get_tool_schema(result.tool_name)
        if tool_schema:
            # Check required params
            required_params = {p.name for p in tool_schema.parameters if p.required}
            provided_params = set(result.params.keys())
            missing_params = required_params - provided_params
            
            # Check for empty required string params
            empty_params = []
            for param in tool_schema.parameters:
                if param.required and param.type == 'string':
                    value = result.params.get(param.name)
                    if not value or (isinstance(value, str) and value.strip() == ""):
                        empty_params.append(param.name)
            
            if missing_params or empty_params:
                logger.debug(
                    f"Validation failed for {result.tool_name}: "
                    f"missing={missing_params}, empty={empty_params}"
                )
                return None  # Agent will retry with better planning
        
        return result
        
    except Exception as e:
        logger.debug(f"Failed to extract planned execution: {e}")
        return None


def _extract_with_base_model(scratchpad: str, tool_name: str) -> ToolExecutionPlanBase | None:
    """Fallback: extract with base model if tool schema not found."""
    # Get secrets from secrets store
    env_vars = _get_all_env_vars()
    env_vars_section = ""
    if env_vars:
        # Escape curly braces in env var values to prevent LangChain template parsing
        def escape_braces(value: str) -> str:
            """Escape curly braces for LangChain template."""
            return str(value).replace("{", "{{").replace("}", "}}")
        
        env_vars_list = "\n".join(
            f"- {key}: {escape_braces(value)}" 
            for key, value in sorted(env_vars.items())
        )
        env_vars_section = f"""

**AVAILABLE ENVIRONMENT VARIABLES:**
The following environment variables are available and can be used for tool parameters:
{env_vars_list}

**IMPORTANT:** If a required parameter matches an env var name or pattern, use the env var value directly.
"""
    
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Extract tool execution plan from the agent's thinking.

If the agent is planning to call execute_tool, extract:
1. tool_name: The exact tool name
2. reasoning: Why this tool is needed
3. params: All parameters as JSON object
4. param_reasoning: Reasoning for each parameter (key: param_name, value: reasoning)
{env_vars_section}

Return structured output matching ToolExecutionPlanBase schema."""),
        ("human", "{scratchpad}")
    ])
    
    try:
        extractor_llm = _get_extractor_llm()
        chain = extraction_prompt | extractor_llm.with_structured_output(
            ToolExecutionPlanBase,
            method="function_calling"
        )
        result = chain.invoke({"scratchpad": scratchpad})
        return result
    except Exception as e:
        logger.debug(f"Failed to extract with base model: {e}")
        return None

