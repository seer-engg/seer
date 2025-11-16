import json
from typing import Dict, List, Optional
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ConfigDict

from agents.eval_agent.constants import (
    LLM,
    N_TEST_CASES,
)
from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from shared.resource_utils import format_resource_hints
from shared.tool_catalog import ToolEntry
from shared.schema import (
    DatasetExample,
)
from shared.tools import (
    LANGCHAIN_MCP_TOOLS,
    think,
)

logger = get_logger("eval_agent.plan.generate_evals")


def _fetch_available_tools(tool_entries: Dict[str, ToolEntry], available_tool_names: List[str]) -> List[ToolEntry]:
    """
    Formats the tool schemas (which are dicts) into a string for the prompt.
    This is the 'explicit contract' for the LLM.
    """
    tools = []
    for name in available_tool_names:
        entry = tool_entries.get(name.lower())
        if entry:
            tools.append(entry)
    return tools


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
{available_tools}

**3. YOUR TASK:**
Generate {n_tests} `DatasetExample` objects.
You MUST use the following iterative process:

1.  **Reason:** Use `think` to analyze the context. What is the biggest weakness? What is a novel attack? (e.g., "I will test a race condition by creating two PRs that link to the same Asana task and see if the agent syncs both or gets confused.")
2.  **Plan Action Steps:** Use `think` to outline the *sequence* of `ActionStep` objects for your test.
    * What tool? (e.g., `asana.create_task`)
    * What params? (e.g., `{{"name": "Race Condition Test Task"}}`)
    * What assertion? (e.g., `asana.get_task` -> `assert_field: "name"`, `assert_expected: "Race Condition Test Task"`)
3.  **Refine Action Steps:**
    * Your `tool` names in the `ActionStep` objects MUST be from the list of available tools.
    * `params` MUST be a valid JSON string.
    * `params` MUST include all `Required Params` for that tool (as defined in your "Contract" above).
4.  **Ground the Test Case:** Once you have a solid list of `ActionStep` objects, create the final `DatasetExample` with:
    * `reasoning`: Why this test is valuable and what it targets.
    * `input_message`: A *plausible* human request that would trigger this behavior (e.g., "Please sync my new PRs.")
    * `expected_output`: An `ExpectedOutput` object containing your list of `ActionStep` objects.
    * `status`: "active"
5.  **Repeat:** Repeat this process until you have {n_tests} high-quality, *different* test cases.
6.  **Final Output:** Your final response MUST be formatted as the `_AgentTestGenerationOutput` object, containing your list of generated tests.
"""

async def _invoke_agentic_llm(
    raw_request: str,
    reflections_text: str,
    prev_dataset_examples: str, 
    agent_name: str,
    user_id: str,
    available_tool_names: List[str],
    tool_entries: Dict[str, ToolEntry],
    resource_hints: str,
    n_tests: int,
) -> List[DatasetExample]:
    
    logger.info(f"plan.test-llm: Starting agentic test generation for {n_tests} tests...")

    # Tools for the generator agent to *plan*
    agent_tools = [think] + LANGCHAIN_MCP_TOOLS

    available_tools = _fetch_available_tools(tool_entries, available_tool_names)
    
    system_prompt = AGENTIC_GENERATOR_SYSTEM_PROMPT.format(
        n_tests=n_tests,
        system_goal_description=raw_request,
        reflections_text=reflections_text,
        prev_dataset_examples=prev_dataset_examples,
        resource_hints=resource_hints,
        available_tools=available_tools,
    )

    # Use the constants LLM
    test_generation_agent = create_agent(
        model=LLM, 
        tools=agent_tools,
        system_prompt=system_prompt,
        state_schema=EvalAgentPlannerState, # Re-use existing state
        response_format=_AgentTestGenerationOutput,
    )
    
    logger.info("Invoking agentic test generator...")
    
    try:
        result = await test_generation_agent.ainvoke(
            {"messages": [HumanMessage(content=f"Please generate {n_tests} complex test cases.")]},
            config=RunnableConfig(recursion_limit=50),
        )
        
        output: _AgentTestGenerationOutput = result.get("structured_response")
        
        # Assign IDs
        for example in output.dataset_examples:
            if not example.example_id:
                example.example_id = str(uuid4())
            example.status = "active"
            
        return output.dataset_examples[:n_tests]
        
    except Exception as e:
        logger.error(f"Agentic test generator failed: {e}. Falling back to genetic method.")
        # Fallback to one empty new test to avoid crashing
        raise e

async def agentic_eval_generation(state: EvalAgentPlannerState) -> dict:
    agent_name = state.github_context.agent_name

    if not state.user_context or not state.user_context.user_id:
        raise ValueError("UserContext with user_id is required to plan")
    user_id = state.user_context.user_id  
    # Get just the inputs from the most recent run
    previous_inputs = [res.dataset_example.input_message for res in state.latest_results]
    
    resource_hints = format_resource_hints(state.mcp_resources)

    logger.info("Using 'agentic' (create_agent) test generation.")
    dataset_examples = await _invoke_agentic_llm(
        raw_request=state.user_context.raw_request,
        reflections_text=state.reflections_text,
        prev_dataset_examples=json.dumps(previous_inputs, indent=2),
        agent_name=agent_name,
        user_id=user_id,
        available_tool_names=state.available_tools,
        tool_entries = state.tool_entries,
        resource_hints=resource_hints,
        n_tests=N_TEST_CASES,
    )

    logger.info("plan.generate: produced %d tests (agent=%s)", len(dataset_examples), agent_name)
    return {
        "dataset_examples": dataset_examples,
    }
