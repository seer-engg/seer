import json
from typing import Dict, List
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ConfigDict

from agents.eval_agent.constants import (
    N_TEST_CASES,
)
from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from shared.resource_utils import format_resource_hints
from shared.schema import (
    DatasetExample,
)
from shared.llm import get_llm

logger = get_logger("eval_agent.plan.generate_evals")

class EvalGenerationOutput(BaseModel):
    dataset_examples: List[DatasetExample] = Field(..., description="The generated test cases")
    model_config = ConfigDict(extra="forbid")


from shared.prompt_loader import load_prompt

# Load prompts from YAML
_PROMPT_CONFIG = load_prompt("eval_agent/generator.yaml")
AGENTIC_GENERATOR_SYSTEM_PROMPT = _PROMPT_CONFIG.system
USER_PROMPT = _PROMPT_CONFIG.user_template



async def _invoke_agentic_llm(
    raw_request: str,
    reflections_text: str,
    prev_dataset_examples: str, 
    resource_hints: str,
    n_tests: int,
) -> List[DatasetExample]:
    
    logger.info("plan.test-llm: Starting test generation for %d tests...", n_tests)    
    
    try:
        
        structured_llm = get_llm(
            model="gpt-4.1",
            temperature=0.0,
        ).with_structured_output(EvalGenerationOutput, method="json_schema", strict=True)

        user_prompt = USER_PROMPT.format(
            n_tests=n_tests,
            system_goal_description=raw_request,
            reflections_text=reflections_text,
            prev_dataset_examples=prev_dataset_examples,
            resource_hints=resource_hints,
        )
        
        # Single LLM call with system prompt + user message
        output: EvalGenerationOutput = await structured_llm.ainvoke([
            SystemMessage(content=AGENTIC_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        
        dataset_examples = []
        for example in output.dataset_examples:
            example.example_id = str(uuid4())
            example.status = "active"
            
            dataset_examples.append(example)
            
        return dataset_examples[:n_tests]
        
    except Exception as e:
        logger.error("Agentic test generator failed: %s", e)
        raise e

async def agentic_eval_generation(state: EvalAgentPlannerState) -> dict:
    agent_name = state.context.github_context.agent_name
    
    # Get just the inputs from the most recent run
    previous_inputs = [res.dataset_example.input_message for res in state.latest_results]
    
    resource_hints = format_resource_hints(state.context.mcp_resources)

    logger.info("Using 'agentic' (structured output) test generation.")
    dataset_examples = await _invoke_agentic_llm(
        raw_request=state.context.user_context.raw_request,
        reflections_text=state.reflections_text,
        prev_dataset_examples=json.dumps(previous_inputs, indent=2),
        resource_hints=resource_hints,
        n_tests=N_TEST_CASES,
    )

    logger.info("plan.generate: produced %d tests (agent=%s)", len(dataset_examples), agent_name)
    return {
        "dataset_examples": dataset_examples,
    }
