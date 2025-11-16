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
from shared.tools.schema_formatter import format_tool_schemas_for_llm
from shared.schema import (
    DatasetExample,
)

logger = get_logger("eval_agent.plan.generate_evals")


class _AgentTestGenerationOutput(BaseModel):
    """The final output from the test generation agent."""
    model_config = ConfigDict(extra="forbid")
    dataset_examples: List[DatasetExample] = Field(
        description=f"A list of {N_TEST_CASES} generated test cases."
    )

AGENTIC_GENERATOR_SYSTEM_PROMPT = """### PROMPT: TEST_CASE_GENERATOR_AGENT (EVAL_AGENT) ###
You are an expert adversarial QA agent. Your goal is to generate {n_tests} new, robust, and creative test cases (`DatasetExample`) to test a target agent.

**1. CONTEXT:**
* **System Goal:** {system_goal_description}
* **Agent Weaknesses (from past reflections):** {reflections_text}
* **Recently Run Tests (Do Not Repeat):** {prev_dataset_examples}
* **Available Resources:** {resource_hints}

**2. AVAILABLE TOOLS & SCHEMAS (Your "Contract"):**
Here is the "contract" for each tool. Your generated actions MUST adhere to these schemas.
{formatted_tool_schemas}

**3. YOUR TASK:**
Generate {n_tests} `DatasetExample` objects.

For each test case:
1.  **Reason:** Analyze the context. What is the biggest weakness? What is a novel attack? (e.g., "Test a race condition by creating two PRs that link to the same Asana task to see if the agent syncs both or gets confused.")
2.  **Design Action Steps:** Plan the *sequence* of `ActionStep` objects for your test.
    * What tool? (must be from the available tools list)
    * What params? (must be valid JSON with all required fields)
    * What assertion? (e.g., check if a field equals expected value)
3.  **Create the Test Case:** Build a `DatasetExample` with:
    * `example_id`: Leave as empty string "" (will be auto-generated)
    * `reasoning`: Why this test is valuable and what it targets.
    * `input_message`: A *plausible* human request that would trigger this behavior (e.g., "Please sync my new PRs.")
    * `expected_output`: An `ExpectedOutput` object containing your list of `ActionStep` objects.
    * `status`: "active"

**Requirements:**
- Tool names MUST match exactly from the available tools list
- Params MUST be valid JSON strings with all required fields
- Generate {n_tests} diverse, high-quality test cases
- Return as `_AgentTestGenerationOutput` with your `dataset_examples` list
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
    
    logger.info(f"plan.test-llm: Starting test generation for {n_tests} tests...")    
    # Format tool schemas using shared formatter
    formatted_tool_schemas = format_tool_schemas_for_llm(tool_entries, available_tool_names)
    
    system_prompt = AGENTIC_GENERATOR_SYSTEM_PROMPT.format(
        n_tests=n_tests,
        system_goal_description=raw_request,
        reflections_text=reflections_text,
        prev_dataset_examples=prev_dataset_examples,
        resource_hints=resource_hints,
        formatted_tool_schemas=formatted_tool_schemas,
    )

    # Use structured output directly (no ReAct agent to avoid thought loops)
    logger.info("Invoking test generator with structured output...")
    
    try:
        # Create a structured LLM that returns _AgentTestGenerationOutput directly
        structured_llm = LLM.with_structured_output(_AgentTestGenerationOutput)
        
        # Single LLM call with system prompt + user message
        output: _AgentTestGenerationOutput = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please generate {n_tests} test cases.")
        ])
        
        # Assign IDs and ensure status
        for example in output.dataset_examples:
            # Generate UUID if example_id is missing or empty
            if not example.example_id or example.example_id == "":
                example.example_id = str(uuid4())
            example.status = "active"
            
        return output.dataset_examples[:n_tests]
        
    except Exception as e:
        logger.error(f"Agentic test generator failed: {e}. Falling back to genetic method.")
        # Fallback to one empty new test to avoid crashing
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
