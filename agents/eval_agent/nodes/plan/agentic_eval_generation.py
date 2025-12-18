import json
from typing import Dict, List
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, ConfigDict
from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from shared.schema import (
    DatasetExample,
)
from shared.llm import get_llm
from shared.config import config
from shared.prompt_loader import load_prompt
import uuid

logger = get_logger("eval_agent.plan.generate_evals")

class EvalGenerationOutput(BaseModel):
    dataset_examples: List[DatasetExample] = Field(..., description="The generated test cases")
    model_config = ConfigDict(extra="forbid")



_PROMPT_CONFIG = load_prompt("eval_agent/generator.yaml")
AGENTIC_GENERATOR_SYSTEM_PROMPT = _PROMPT_CONFIG.system
USER_PROMPT = _PROMPT_CONFIG.user_template
ALL_PROMPTS = _PROMPT_CONFIG.all_prompts


async def agentic_eval_generation(state: EvalAgentPlannerState) -> dict:
    agent_name = state.context.agent_name
    
    # Get just the inputs from the most recent run
    previous_inputs = [res.dataset_example.input_message for res in state.latest_results]

    raw_request = state.context.user_context.raw_request
    reflections_text = state.reflections_text
    prev_dataset_examples = json.dumps(previous_inputs, indent=2)
    mcp_services = state.context.mcp_services

    resource_hints = state.context.mcp_resources

    logger.info("Using 'agentic' (structured output) test generation.")

    # Get reasoning effort from config (can be overridden via env var)
    reasoning_effort = config.eval_reasoning_effort
    logger.info(f"Using reasoning_effort: {reasoning_effort}")

    structured_llm = get_llm(
            reasoning_effort=reasoning_effort,
        ).with_structured_output(EvalGenerationOutput, method="json_schema", strict=True)

    user_prompt = USER_PROMPT.format(
        n_tests=config.eval_n_test_cases,
        system_goal_description=raw_request,
        reflections_text=reflections_text,
        prev_dataset_examples=prev_dataset_examples,
        resource_hints=resource_hints,
        mcp_services=mcp_services,
    )

    SYSTEM_PROMPT = AGENTIC_GENERATOR_SYSTEM_PROMPT.format(resource_hints=resource_hints)

    for service in mcp_services:
        SYSTEM_PROMPT += ALL_PROMPTS[service]
    
    # Single LLM call with system prompt + user message
    output: EvalGenerationOutput = await structured_llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])
    
    dataset_examples = []
    for example in output.dataset_examples:
        example.example_id = str(uuid4())
        example.status = "active"
        
        dataset_examples.append(example)
        
    dataset_examples = dataset_examples[:config.eval_n_test_cases]

    logger.info("plan.generate: produced %d tests (agent=%s)", len(dataset_examples), agent_name)
    output_messages = [ToolMessage(content=dataset_examples, tool_call_id=str(uuid.uuid4()))]
    return {
        "dataset_examples": dataset_examples,
        "messages": output_messages,
    }
