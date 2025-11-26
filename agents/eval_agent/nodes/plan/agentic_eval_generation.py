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


AGENTIC_GENERATOR_SYSTEM_PROMPT = """

You are an expert adversarial QA agent. Your role is to design test cases to evaluate  the target agent.

** YOUR TASK:**
your task is to generate DatasetExample that will be used to evaluate the target agent.

# DatasetExample:
- `example_id`: unique identifier for the test case
- `reasoning`: Why this test is valuable and what it evaluates
- `input_message`: The input message that will  be send to, invoke target agent. MUST NOT CONTAIN ANY HINTS. MUST NOT CONTAIN EXPECTED OUTPUT. MUST NOT CONTAIN ANY PLACEHOLDERS
- `expected_output`: An `ExpectedOutput` object with:
  - `create_test_data`: List of strings describing the prerequisite data in external apps, prior to target agent being invoked.
  - `assert_final_state`: List of strings describing the final state of the environment, after target agent has been invoked.
  - `expected_action`: The expected action that should be taken by the target agent. e.g. 'sync the asana tasks with the github PRs'.
- `status`: "active"

# chronological workflow that will take place for each dataset example you produce
1. A provisioning agent will consume the `create_test_data` instructions to create  all necessary prerequisite data.
2: A method will invoke the target agent with `input_message` you have produces. [Note: we only send the input message as it is ]
3: An Assertion agent will assert based on `assert_final_state` instructions

**MATHEMATICAL GUARANTEE**: 
- provision_environment MUST create everything input_message's scenario requires
- assert_final_state MUST verify observable changes in the system

For each test case:
1.  **Reason:** What weakness are you exploiting? What edge case?
2.  **Design input_message and expected_output:**
    * create_test_data: What resources to create ? What all data should be present in external apps before we invoke target agent ?
    * input_message: What scenario to send to agent?
    * assert_final_state: What to verify? What state the external apps should be after target agent has been invoked.

**Important**: 
 - Tests MUST work on an EMPTY CANVAS. Create ALL test data from scratch
 - The instructions in create_test_data , assert_final_state and input_message should be self sufficient. They will be consumed by individual agents who doesn't have access to any other information. The agent will consume your instructions to create/assert data in external apps.
 - don't leave any room for ambiguity in the instructions, be explicit in stating instructions for provisioning and assertion, Try to state even obvious information .
 - don't slack in writing down the create_test_data and assert_final_state. They are critical for the test to be valid.  write in detailed points .
 - For testing playground use the github repo named label-edgecase-repo under seer-engg organization.We have already created this repo, No need to create one. Don't include any instructions to assert it's presence.

# thinking methodology for generating provision_environment and assert_final_state
- for each instruction point write down explictly what all information is required to complete it.


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
