import json
from typing import Dict, List
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ConfigDict

from agents.eval_agent.constants import (
    LLM,
    N_TEST_CASES,
)
from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from shared.resource_utils import format_resource_hints
from shared.tools import ToolEntry
from shared.schema import (
    DatasetExample,
)

logger = get_logger("eval_agent.plan.generate_evals")

class EvalGenerationOutput(BaseModel):
    dataset_examples: List[DatasetExample] = Field(..., description="The generated test cases")
    model_config = ConfigDict(extra="forbid")


AGENTIC_GENERATOR_SYSTEM_PROMPT = """
You are an expert adversarial QA agent. Your role is to design test cases for the target agent.


** YOUR TASK:**
your task is to generate DatasetExample objects for the target agent:

# DatasetExample:
- `example_id`: unique identifier for the test case
- `reasoning`: Why this test is valuable and what it targets
- `input_message`: The input message that should be send to target agent. MUST NOT CONTAIN ANY HINTS. MUST NOT CONTAIN EXPECTED OUTPUT!
- `expected_output`: An `ExpectedOutput` object with:
  - `provision_environment`: List of strings describing the prerequisite state of the environment, prior to target agent being invoked.
  - `assert_final_state`: List of strings describing the final state of the environment, after target agent has been invoked.
- `status`: "active"

**MATHEMATICAL GUARANTEE**: 
- provision_environment MUST create everything input_message's scenario requires
- assert_final_state MUST verify observable changes in the system
- NEVER assume existing data exists

For each test case:
1.  **Reason:** What weakness are you exploiting? What edge case?
2.  **Design input_message and expected_output:**
    * provision_environment: What resources to create? What state the environment should be in prior to target agent being invoked.
    * input_message: What scenario to send to agent?
    * assert_final_state: What to verify? What state the environment should be in after target agent has been invoked.

**CRITICAL**: 
 - Tests MUST work on an EMPTY CANVAS. Create ALL test data from scratch
 - The instructions in provision_environment and assert_final_state should be self sufficient. They will be consumed by individual agents who doesn't have access to any other information. So both provision_environment and assert_final_state should be detailed enough to be consumed by the agents.
 - don't leave any room for ambiguity in the instructions for example don't say create a repo named 'test-repo' instead say create a repo named 'test-repo' in organization 'seer-engg'.
 - don't slack in writing down the provision_environment and assert_final_state. They are critical for the test to be valid.  write in detailed points for provision_environment and assert_final_state.
"""

USER_PROMPT = """
Based on the context of the target agent, generate {n_tests} test cases for the target agent.
** CONTEXT of Target Agent:**
* **System Goal:** {system_goal_description}
* **Agent Weaknesses (from past reflections):** {reflections_text}
* **Recently Run Tests (Do Not Repeat):** {prev_dataset_examples}
* **Available Resources:** {resource_hints}
"""


async def _invoke_agentic_llm(
    raw_request: str,
    reflections_text: str,
    prev_dataset_examples: str, 
    available_tool_names: List[str],
    tool_entries: Dict[str, ToolEntry],
    resource_hints: str,
    n_tests: int,
) -> List[DatasetExample]:
    
    logger.info("plan.test-llm: Starting test generation for %d tests...", n_tests)    
    
    try:
        
        structured_llm = LLM.with_structured_output(EvalGenerationOutput, method="json_schema", strict=True)

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
        available_tool_names=state.available_tools,
        tool_entries=state.tool_entries,
        resource_hints=resource_hints,
        n_tests=N_TEST_CASES,
    )

    logger.info("plan.generate: produced %d tests (agent=%s)", len(dataset_examples), agent_name)
    return {
        "dataset_examples": dataset_examples,
    }
