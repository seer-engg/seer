"""Verify provisioning succeeded before invoking target agent.

This evaluates the eval agent's ability to follow its own plan in isolation,
independent of target agent execution quality.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.llm import get_llm
from langchain_core.messages import HumanMessage
from shared.tools import ComposioMCPClient
from shared.config import config
from .utils import handle_tool_errors
from langchain.agents import create_agent
from shared.prompt_loader import load_prompt

logger = get_logger("eval_agent.execute.verify_provisioning")

# Load prompts
_PROMPT_CONFIG = load_prompt("eval_agent/assert.yaml")
SYSTEM_PROMPT = _PROMPT_CONFIG.system
USER_PROMPT = _PROMPT_CONFIG.user_template


class ProvisioningVerification(BaseModel):
    """LLM-as-judge verification of provisioning success."""
    provisioning_succeeded: bool = Field(description="Whether provisioning succeeded according to the plan")
    verification_reasoning: str = Field(description="Detailed explanation of why provisioning succeeded or failed")
    missing_requirements: List[str] = Field(default_factory=list, description="List of requirements from create_test_data that were not met")
    unexpected_changes: List[str] = Field(default_factory=list, description="List of unexpected changes or side effects")


async def verify_provisioning_node(state: TestExecutionState) -> dict:
    """
    Verify that provisioning succeeded according to the plan's create_test_data instructions.
    
    This is a checkpoint BEFORE target agent invocation. We evaluate the eval agent's
    ability to follow its own plan in isolation.
    """
    example = state.dataset_example
    if not example:
        raise ValueError("verify_provisioning_node requires dataset_example in state")
    
    provisioning_output = state.provisioning_output or ""
    
    # Extract create_test_data instructions (what should have been created)
    create_instructions: List[str] = []
    if example.expected_output:
        for service_instructions in example.expected_output.create_test_data:
            create_instructions.extend(service_instructions.instructions)
    
    if not create_instructions:
        # No provisioning required - consider it succeeded
        logger.info("No create_test_data instructions - provisioning verification skipped")
        return {
            "provisioning_verification": {
                "provisioning_succeeded": True,
                "verification_reasoning": "No provisioning required",
                "missing_requirements": [],
                "unexpected_changes": []
            }
        }
    
    # Use LLM-as-judge to verify provisioning succeeded
    verification_prompt = f"""You are verifying that provisioning succeeded according to the plan.

PLAN REQUIREMENTS (from create_test_data):
{chr(10).join(f"- {inst}" for inst in create_instructions)}

PROVISIONING OUTPUT (what the provisioning agent actually did):
{provisioning_output}

Your task: Verify whether the provisioning agent successfully created all required test data according to the plan.

Check:
1. Were all required resources created?
2. Do they match the specifications in create_test_data?
3. Are there any missing requirements?
4. Are there any unexpected side effects or changes?

Use the tools available to you to inspect the actual state of the environment and compare it to the plan requirements."""

    # Get tools for verification
    # TODO: replace hardcoded asana services
    tool_service = ComposioMCPClient(["GITHUB", "ASANA"], state.context.user_id)
    all_tools = await tool_service.get_tools()
    
    actual_tools = []
    for tool in all_tools:
        if tool.name in state.context.tool_entries.keys():
            actual_tools.append(tool)
    
    # Use high reasoning effort for verification (verification should be thorough)
    verification_agent = create_agent(
        model=get_llm(model='gpt-5-mini'),  # Hardcoded to 'high' for thorough verification
        tools=actual_tools,
        system_prompt="You are a verification agent. Your job is to verify that provisioning succeeded by comparing the actual environment state to the plan requirements. Use tools to inspect the environment and provide a clear verdict.",
        middleware=[handle_tool_errors]
    )
    
    user_prompt = HumanMessage(content=verification_prompt)
    
    try:
        result = await verification_agent.ainvoke(input={"messages": [user_prompt]})
        verification_output = result.get('messages')[-1].content
        
        # Use structured LLM to parse verification result
        verification_llm = get_llm(
            model="gpt-5-mini",
            temperature=0.0,
        ).with_structured_output(ProvisioningVerification, method="function_calling", strict=True)
        
        # Create a summary prompt for structured parsing
        summary_prompt = f"""Based on the verification output below, determine if provisioning succeeded.

VERIFICATION OUTPUT:
{verification_output}

PLAN REQUIREMENTS:
{chr(10).join(f"- {inst}" for inst in create_instructions)}

Provide a structured verdict on whether provisioning succeeded."""
        
        verification_result = await verification_llm.ainvoke([HumanMessage(content=summary_prompt)])
        
        logger.info(f"Provisioning verification: {'✅ SUCCEEDED' if verification_result.provisioning_succeeded else '❌ FAILED'}")
        if not verification_result.provisioning_succeeded:
            logger.warning(f"Missing requirements: {verification_result.missing_requirements}")
            logger.warning(f"Unexpected changes: {verification_result.unexpected_changes}")
        
        return {
            "provisioning_verification": {
                "provisioning_succeeded": verification_result.provisioning_succeeded,
                "verification_reasoning": verification_result.verification_reasoning,
                "missing_requirements": verification_result.missing_requirements,
                "unexpected_changes": verification_result.unexpected_changes,
                "verification_output": verification_output
            }
        }
        
    except Exception as e:
        logger.error(f"Error during provisioning verification: {e}")
        # On error, assume failure to be safe
        return {
            "provisioning_verification": {
                "provisioning_succeeded": False,
                "verification_reasoning": f"Verification failed with error: {str(e)}",
                "missing_requirements": ["Verification could not complete"],
                "unexpected_changes": [],
                "verification_output": str(e)
            }
        }

