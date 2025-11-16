"""
Layer 2: Parameter Inference

BARBELL STRATEGY:
- Left: Exact matches (deterministic, fast)
- Right: LLM inference (intelligent, slower)
- NO MIDDLE: No fuzzy logic, no heuristics, no maintenance

Design: Simple and powerful - let the LLM handle intelligence.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from shared.logger import get_logger
from shared.config import DEFAULT_LLM_MODEL

logger = get_logger("parameter_population.parameter_inference")

# Fast LLM for parameter inference
_llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0)


class ParameterInferenceResult(BaseModel):
    """LLM output for parameter inference."""
    context_var_name: Optional[str] = None
    reasoning: str


async def infer_parameter_value(
    param_name: str,
    tool_name: str,
    context_vars: Dict[str, Any],
    param_schema: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    BARBELL STRATEGY: Exact match OR LLM inference. No fuzzy logic.
    
    Left (Fast): Exact matching
    Right (Smart): LLM structured inference
    
    Args:
        param_name: Name of the parameter to infer
        tool_name: Full tool name (e.g., "github_list_pull_requests")
        context_vars: Available context variables
        param_schema: JSON schema for the parameter (if available)
        
    Returns:
        Inferred value or None if no match found
    """
    # LEFT SIDE: Exact matches only (deterministic)
    
    # 1. Direct exact match
    if param_name in context_vars:
        logger.debug("Exact match: %s", param_name)
        return context_vars[param_name]
    
    # 2. Service convention: {service}_{param_name}
    service = _extract_service_from_tool(tool_name)
    convention_key = f"{service}_{param_name}"
    if convention_key in context_vars:
        logger.debug("Convention match: %s → %s", param_name, convention_key)
        return context_vars[convention_key]
    
    # RIGHT SIDE: LLM inference (intelligent)
    logger.info("No exact match for '%s', using LLM inference...", param_name)
    return await _llm_infer_parameter(
        param_name=param_name,
        param_schema=param_schema,
        context_vars=context_vars,
        tool_name=tool_name,
        service=service,
    )


def _extract_service_from_tool(tool_name: str) -> str:
    """Extract service name from tool name."""
    if '_' in tool_name:
        return tool_name.split('_')[0]
    if '.' in tool_name:
        return tool_name.split('.')[0]
    return 'unknown'


async def _llm_infer_parameter(
    param_name: str,
    param_schema: Optional[Dict[str, Any]],
    context_vars: Dict[str, Any],
    tool_name: str,
    service: str,
) -> Optional[Any]:
    """
    Use LLM to intelligently infer which context variable should map to this parameter.
    
    This is the RIGHT side of the barbell - intelligent, maintainable, powerful.
    """
    if not context_vars:
        return None
    
    # Build param description
    param_desc = ""
    if param_schema:
        param_desc = param_schema.get('description', '')
        param_type = param_schema.get('type', 'unknown')
        param_desc = f"Type: {param_type}. {param_desc}" if param_desc else f"Type: {param_type}"
    
    # Format context vars for LLM
    context_list = "\n".join([f"- {k}: {v}" for k, v in context_vars.items()])
    
    system_prompt = """You are a parameter mapping expert. Your job is to match a required parameter to the correct context variable.

Rules:
1. Only return a context_var_name if you're CONFIDENT it's the right match
2. If no good match exists, return null for context_var_name
3. Be strict - wrong matches are worse than no matches
4. Consider the service, parameter name, and descriptions"""

    user_prompt = f"""Match this parameter to a context variable:

Tool: {tool_name}
Service: {service}
Parameter: {param_name}
{f"Description: {param_desc}" if param_desc else ""}

Available context variables:
{context_list}

Which context variable should map to '{param_name}'?"""

    try:
        structured_llm = _llm.with_structured_output(ParameterInferenceResult)
        result: ParameterInferenceResult = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        if result.context_var_name and result.context_var_name in context_vars:
            logger.info(
                "LLM matched '%s' → '%s' (reason: %s)", 
                param_name, 
                result.context_var_name,
                result.reasoning
            )
            return context_vars[result.context_var_name]
        else:
            logger.warning(
                "LLM could not find match for '%s' (reason: %s)",
                param_name,
                result.reasoning
            )
            return None
            
    except Exception as e:
        logger.error("LLM inference failed for parameter '%s': %s", param_name, e)
        return None


def get_missing_required_parameters(
    current_params: Dict[str, Any],
    required_params: List[str],
) -> List[str]:
    """
    Determine which required parameters are missing or empty.
    
    Args:
        current_params: Currently populated parameters
        required_params: List of required parameter names from schema
        
    Returns:
        List of missing parameter names
    """
    missing = []
    for param in required_params:
        if param not in current_params:
            missing.append(param)
        elif current_params[param] is None or current_params[param] == "":
            missing.append(param)
    
    return missing

