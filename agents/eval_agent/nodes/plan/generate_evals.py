import asyncio
import json
import os  # ADDED
from typing import Any, Dict, List
from uuid import uuid4

from langchain.agents import create_agent  # ADDED
from langchain_core.messages import HumanMessage, SystemMessage  # ADDED
from langchain_core.runnables import RunnableConfig  # ADDED
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ConfigDict

from agents.eval_agent.constants import (  # MODIFIED
    LLM,
    NEO4J_GRAPH,
    N_TEST_CASES,
)
from agents.eval_agent.models import EvalAgentState, ToolSelectionLog
from agents.eval_agent.reflection_store import graph_rag_retrieval
from shared.logger import get_logger
from shared.resource_utils import format_resource_hints
from shared.schema import (  # ADDED
    ActionStep,
    DatasetExample,
    ExpectedOutput,
)
from shared.tool_catalog import (
    ToolEntry,
    build_tool_name_set,
    canonicalize_tool_name,
    load_tool_entries,
    select_relevant_tools,
)
from shared.tools import (  # ADDED
    LANGCHAIN_MCP_TOOLS,
    think,
    web_search,
)

logger = get_logger("eval_agent.plan.generate_evals")


class _TestGenerationOutput(BaseModel):
    """Helper for structured output"""
    model_config = ConfigDict(extra="forbid")
    dataset_example: DatasetExample


def _validate_generated_actions(
    examples: List[DatasetExample], tool_entries: Dict[str, ToolEntry]
) -> None:
    """Ensure every generated action references a known tool."""

    if not tool_entries:
        return

    name_map = build_tool_name_set(tool_entries)
    valid_names = set(name_map.keys())
    valid_names.add("system.wait")

    invalid: List[str] = []
    for example in examples:
        # ADDED: Handle cases where expected_output might be None
        if not example.expected_output:
            invalid.append(
                f"example={example.example_id or '<pending>'} has missing expected_output"
            )
            continue
        for idx, action in enumerate(example.expected_output.actions):
            normalized = canonicalize_tool_name(action.tool)
            if normalized not in valid_names:
                invalid.append(
                    f"example={example.example_id or '<pending>'} action_index={idx} tool={action.tool}"
                )

    if invalid:
        sample = ", ".join(list(name_map.values())[:20])
        raise ValueError(
            "Generated actions referenced unknown tools or had missing data. Details: "
            + ", ".join(invalid)
            + (f". Known tools include: {sample}" if sample else "")
        )


COMMON_INSTRUCIONS = """\n\n
**Your Task:**
Create one new `DatasetExample` that is a complex, adversarial test case.
Your test case MUST be a JSON list of actions.

**Available Tools:**
You *must* pick your tool names from this list (exact strings such as `ASANA_CREATE_TASK`).
<tools>
{available_tools}
</tools>

**Available Resources:**
Use `[resource:resource_name.field]` tokens when you need stable IDs.
{resource_hints}

**Action Schema (ALL FIELDS REQUIRED):**
Each action in the `expected_output.actions` list is a JSON object with:
- `tool`: (str) The FULL tool name, chosen from the <tools> list (e.g., "ASANA_CREATE_SUBTASK", "GITHUB_CREATE_PULL_REQUEST", "system.wait").
- `params`: (str) A JSON STRING of the parameters for the tool. e.g., "{{\\"name\\": \\"My Task\\"}}" or "{{}}"
- `assign_to_var`: (str) Variable name to store output ID. Use an empty string "" if not needed.
- `assert_field`: (str) A JSON path to check in the tool's output (e.g., "status.name"). Use an empty string "" if not needed.
- `assert_expected`: (str) The expected value, AS A STRING. Use an empty string "" if not needed.

**Rules for Action-Based Tests:**
1.  **Use Available Tools:** The `tool` field must be an *exact match* from the <tools> list.
2.  **System Tools:** For waiting, use `tool: "system.wait"`.
3.  **ALL FIELDS ARE REQUIRED:** You must provide a value for every field. Use "" for empty.
4.  **Params is a JSON String:** The `params` field *must* be a string containing valid JSON.
5.  **Assert Expected is a String:** The `assert_expected` field *must* be a string.
6.  **Environment is Ready:** You MUST assume all required MCP services are already provisioned.
7.  **Variable Usage:** Use `[var:variable_name]` in your `params` JSON string.
8.  **Assertion:** To make an assertion, provide a non-empty `assert_field`.

**Example Test Case (using real tool names):**
{{ "reasoning": "Tests if merging a PR updates the linked Asana task's status.", "input_message": "Please sync GitHub PR merges to Asana task statuses.", "expected_output": {{ "actions": [ {{ "tool": "ASANA_CREATE_SUBTASK", "params": "{{\"name\": \\"Test Task\", \"notes\": \"Ticket for PR sync test\"}}", "assign_to_var": "ticket_id", "assert_field": "", "assert_expected": "" }}, {{ "tool": "GITHUB_CREATE_PULL_REQUEST", "params": "{{\"title\": \\"Test PR\", \"body\": \"Links to [var:ticket_id]\"}}", "assign_to_var": "pr_id", "assert_field": "", "assert_expected": "" }}, {{ "tool": "system.wait", "params": "{{\"seconds\": 30}}", "assign_to_var": "", "assert_field": "", "assert_expected": "" }}, {{ "tool": "ASANA_GET_A_TASK", "params": "{{\"id\": \"[var:ticket_id]\"}}", "assign_to_var": "", "assert_field": "status", "assert_expected": "Done" }} ] }}, "status": "active" }}
Provide your final output as *only* the new `DatasetExample` Pydantic object, matching the schema.
"""



MUTATION_PROMPT = """### PROMPT: TEST_CASE_MUTATOR (EVAL_AGENT) ###
You are an adversarial "Test Case Mutator."
The target agent *passed* this test, which is a *failure* for you. You MUST create a harder test.

**Parent Test (Passed):**
{parent_test_json}

**Past Reflections (Agent Weaknesses):**
{reflections_text}

**Your Task:**
Create one new `DatasetExample` that is a *harder, mutated* version of this parent.
You MUST use the following Chain of Thought:

**Chain of Thought (MANDATORY):**
1.  **Analyze Parent:** What simple action sequence does the parent test? (e.g., 'create task, create PR, check status')
2.  **Brainstorm Mutations:** Brainstorm 3 *different* ways to mutate the parent's action list to make it harder.
    * **Genetic Operators:** `add_noise` (add extra irrelevant actions), `add_edge_case` (use empty strings, special characters), `change_order` (trigger actions in an unexpected order), `add_delay` (add extra `wait` steps).
3.  **Select Best Mutation:** Which of your 3 ideas is *most likely* to cause a sophisticated sync agent to fail?
""" + COMMON_INSTRUCIONS

CROSSOVER_PROMPT = """### PROMPT: TEST_CASE_BREEDER (EVAL_AGENT) ###
You are an adversarial "Test Case Breeder."
You will be given two "fit" parent test cases that each *found a bug*.
Your job is to create one new, "hybrid" test case that combines the "genetic material" (the failure modes) of *both* parents.

**Parent 1 (Fit):**
{parent_1_json}

**Parent 2 (Fit):**
{parent_2_json}

**Past Reflections (Agent Weaknesses):**
{reflections_text}

**Your Task:**
Create one new `DatasetExample` that is a *complex, hybrid* of both parents.
You MUST use the following Chain of Thought:

**Chain of Thought (MANDATORY):**
1.  **Analyze Parents:** What failure mode did Parent 1 find? (e.g., 'failed on `KeyError`'). What failure mode did Parent 2 find? (e.g., 'failed on `ZeroDivisionError`').
2.  **Brainstorm Hybrids:** Brainstorm 2 ways to create a *single piece of buggy code* that could suffer from *both* failure modes (e.g., a function that accesses a dict *and* performs division).
3.  **Select Best Hybrid:** Which of your 2 ideas is the *most complex* and difficult test?
""" + COMMON_INSTRUCIONS

NEW_TEST_PROMPT = """### PROMPT: NEW_TEST_GENERATOR (EVAL_AGENT) ###
You are an adversarial QA analyst. Your goal is to brainstorm *one* novel test case.

**Raw request:**
{raw_request}

**Past Reflections (Agent Weaknesses):**
{reflections_text}

**Recently Run Tests (Do Not Repeat These):**
{prev_dataset_examples}

**Your Task:**
Create one new, creative, and *hard* `DatasetExample`.
**Your primary goal is to make the agent fail.**
DO NOT CREATE SIMPLE TESTS!
You MUST use the following Chain of Thought:

**Chain of Thought (MANDATORY):**
1.  **Analyze Weaknesses:** Based on "Past Reflections," what is the agent's *biggest* known weakness? (If no reflections, focus on the "Raw request").
2.  **Brainstorm New Attack:** Brainstorm 3 *new, different, and complex* test scenarios.
    * **Adversarial Techniques:** - `state_change_race`: Update the *same* resource on two systems at once and see if the agent gets confused or creates a loop.
        - `invalid_data`: Create a PR that links to a *deleted* Asana task.
        - `permission_error`: (Simulated) Create a resource and then make it read-only.
        - `rapid_updates`: Update a PR title 5 times in 5 seconds. Does it spam Asana?
        - `out_of_order`: Comment on a PR *before* linking it to a task.
3.  **Select Best Attack:** - Which of your 3 ideas **combines at least two** adversarial techniques (e.g., `nest` + `add_error_condition`) and is *most likely to cause a failure*? 
    - This attack MUST NOT be in "Recently Run Tests".
""" + COMMON_INSTRUCIONS


# --- RENAMED: This is the old "genetic" / structured prompt method ---
async def _invoke_genetic_llm(
    raw_request: str,
    reflections_text: str,
    prev_dataset_examples: str, # JSON string of recent inputs
    agent_name: str,
    user_id: str,
    available_tools: List[str],
    resource_hints: str,
    force_new_test: bool = False, # ADDED
) -> List[DatasetExample]:
    
    logger.info("plan.test-llm: Starting 'genetic' (structured prompt) test generation...")
    
    # Use a smart, critical LLM for this
    _smart_llm = ChatOpenAI(
            model="gpt-5-codex",
            use_responses_api=True,            
            output_version="responses/v1",     
            reasoning={"effort": "low"},
        )
    test_generation_llm = _smart_llm.with_structured_output(_TestGenerationOutput)

    new_generation: List[DatasetExample] = []
    
    # --- ADDED: Format the tool list for the prompt ---
    tool_names = list(dict.fromkeys(available_tools or []))
    lower_names = {name.lower() for name in tool_names}
    if "system.wait" not in lower_names:
        tool_names.append("system.wait")
    tools_list_str = "\n".join(tool_names) if tool_names else "system.wait"
    # --- END ADD ---

    # --- Define our new "Genetic Operator" functions ---
    
    async def get_parent(passed: bool, k: int = 1) -> List[dict]:
        """Helper to get a parent test from Neo4j."""
        # Find an active test, that was run, and had the desired pass/fail outcome
        cypher_query = """
        MATCH (ex:DatasetExample {user_id: $user_id, agent_name: $agent_name, status: 'active'})
        MATCH (ex)-[:WAS_RUN_IN]->(res:ExperimentResult {passed: $passed})
        RETURN ex.reasoning as reasoning, ex.input_message as input_message, ex.expected_output as expected_output, ex.example_id as example_id, ex.status as status
        LIMIT $k
        """
        result = await asyncio.to_thread(
            NEO4J_GRAPH.query,
            cypher_query,
            params={"user_id": user_id, "agent_name": agent_name, "passed": passed, "k": k}
        )
        # MODIFIED: Parse the expected_output from string back to dict
        parsed_results = []
        for r in result:
            r_dict = dict(r)
            try:
                # expected_output is stored as a JSON string in Neo4j
                # It should contain '{"actions": [...] }'
                r_dict['expected_output'] = json.loads(r_dict['expected_output'])
            except Exception:
                logger.warning(f"Could not parse expected_output for parent test {r_dict.get('example_id')}")
                continue
            parsed_results.append(r_dict)
        return parsed_results

    async def run_crossover(parents: List[dict]):
        """Run the Crossover prompt"""
        logger.info("plan.test-llm: Running Crossover...")
        # --- ADDED available_tools ---
        prompt = CROSSOVER_PROMPT.format(
            parent_1_json=json.dumps(parents[0], indent=2),
            parent_2_json=json.dumps(parents[1], indent=2),
            reflections_text=reflections_text,
            available_tools=tools_list_str,
            resource_hints=resource_hints,
        )
        # --- END ADD ---
        output = await test_generation_llm.ainvoke(prompt)
        new_generation.append(output.dataset_example)

    async def run_mutation(parent: dict):
        """Run the Mutation prompt"""
        logger.info("plan.test-llm: Running Mutation...")
        # --- ADDED available_tools ---
        prompt = MUTATION_PROMPT.format(
            parent_test_json=json.dumps(parent, indent=2),
            reflections_text=reflections_text,
            available_tools=tools_list_str,
            resource_hints=resource_hints,
        )
        # --- END ADD ---
        output = await test_generation_llm.ainvoke(prompt)
        new_generation.append(output.dataset_example)

    async def run_new_test():
        """Run the New Test prompt"""
        logger.info("plan.test-llm: Running New Test generation...")
        # --- ADDED available_tools ---
        prompt = NEW_TEST_PROMPT.format(
            raw_request=raw_request,
            reflections_text=reflections_text,
            prev_dataset_examples=prev_dataset_examples,
            available_tools=tools_list_str,
            resource_hints=resource_hints,
        )
        # --- END ADD ---
        output = await test_generation_llm.ainvoke(prompt)
        new_generation.append(output.dataset_example)

    # --- Genetic Algorithm "Breeding" Strategy ---

    # --- ADDED: Fallback logic ---
    if force_new_test:
        logger.info("Forcing generation of one new test as a fallback.")
        await run_new_test()
    else:
        # --- Original logic ---
        fit_parents = await get_parent(passed=False, k=2) # Get 2 FAILED tests
        unfit_parents = await get_parent(passed=True, k=1) # Get 1 PASSED test
        has_history = bool(fit_parents or unfit_parents)
        
        # --- NEW: Determine how many tests to generate ---
        # If there's no history, generate more tests to increase chance of a failure
        logger.info(f"plan.test-llm: Generating {N_TEST_CASES} tests (has_history={has_history})")
        
        # 1. Crossover (Exploitation)
        if has_history and len(new_generation) < N_TEST_CASES:
            if len(fit_parents) == 2:
                await run_crossover(fit_parents)
            else:
                logger.warning("Could not find 2 'fit' (failed) parents for crossover. Skipping.")

        # 2. Mutation (Exploitation)
        if has_history and len(new_generation) < N_TEST_CASES:
            if unfit_parents:
                await run_mutation(unfit_parents[0])
            else:
                logger.warning("Could not find 1 'unfit' (passed) parent for mutation. Skipping.")

        # 3. New Tests (Exploration)
        # Fill the remaining slots with brand new tests
        while len(new_generation) < N_TEST_CASES:
            await run_new_test()

    # --- Assign final Example IDs ---
    for example in new_generation:
        # ADDED: Ensure example_id is None before assigning new one
        if not getattr(example, 'example_id', None):
            example.example_id = str(uuid4())
        example.status = "active"

    logger.info(
        "plan.test-llm: Evolutionary generation complete. Produced %d tests.",
        len(new_generation),
    )
    
    return new_generation[:N_TEST_CASES]


# --- ADDED: Agentic (create_agent) generation method ---

class _AgentTestGenerationOutput(BaseModel):
    """The final output from the test generation agent."""
    model_config = ConfigDict(extra="forbid")
    dataset_examples: List[DatasetExample] = Field(
        description=f"A list of {N_TEST_CASES} generated test cases."
    )
    generation_summary: str = Field(
        description="A brief summary of the agent's reasoning and strategy (e.g., 'created 1 new test, 1 mutation')."
    )

AGENTIC_GENERATOR_SYSTEM_PROMPT = """### PROMPT: TEST_CASE_GENERATOR_AGENT (EVAL_AGENT) ###
You are an adversarial QA agent. Your goal is to generate {n_tests} new, complex, and creative test cases (`DatasetExample`) to make the target agent fail.

**1. CONTEXT:**
* **Raw Request:** {raw_request}
* **Agent Weaknesses (from past reflections):** {reflections_text}
* **Recently Run Tests (Do Not Repeat):** {prev_dataset_examples}
* **Available Resources:** {resource_hints}

**2. AVAILABLE TOOLS:**
You have access to tools like `think`, `web_search`, and documentation search (`langchain_docs_...`).
You MUST use these tools to iteratively plan and refine your test cases.
* Use `think` to reason about your plan.
* Use `web_search` or `langchain_docs` if you are unsure about tool parameters or behavior.
* Your goal is to *plan* a list of `ActionStep` objects, not to *execute* them (like `asana.create_task`).

**3. YOUR TASK:**
Generate {n_tests} `DatasetExample` objects.
You MUST use the following iterative process:

1.  **Reason:** Use `think` to analyze the context. What is the biggest weakness? What is a novel attack? (e.g., "I will test a race condition by creating two PRs that link to the same Asana task and see if the agent syncs both or gets confused.")
2.  **Plan Action Steps:** Use `think` to outline the *sequence* of `ActionStep` objects for your test.
    * What tool? (e.g., `asana_create_task`)
    * What params? (e.g., `{{"name": "Race Condition Test Task"}}`)
    * Assign to variable? (e.g., `task_id`)
    * What assertion? (e.g., `asana_get_task` -> `assert_field: "name"`, `assert_expected: "Race Condition Test Task"`)
3.  **Refine Action Steps:**
    * Your `tool` names in the `ActionStep` objects MUST be *canonicalized* (e.g., `asana_create_task`, `github_create_pull_request`, `system_wait`).
    * This is a *suggested* list of canonical names: {available_tool_names}
    * `params` MUST be a JSON string.
    * `assign_to_var`, `assert_field`, `assert_expected` MUST be provided (use "" if empty).
4.  **Ground the Test Case:** Once you have a solid list of `ActionStep` objects, create the final `DatasetExample` with:
    * `reasoning`: Why this test is valuable and what it targets.
    * `input_message`: A *plausible* human request that would trigger this behavior (e.g., "Please sync my new PRs.")
    * `expected_output`: An `ExpectedOutput` object containing your list of `ActionStep` objects.
    * `status`: "active"
    * `example_id`: null (the system will assign this)
5.  **Repeat:** Repeat this process until you have {n_tests} high-quality, *different* test cases.
6.  **Final Output:** Your final response MUST be formatted as the `_AgentTestGenerationOutput` object, containing your list of generated tests.

**RULES:**
* **DO NOT** just copy a test from the "Recently Run" list.
* **DO** create complex, multi-step tests that target *interactions* (e.g., Asana + GitHub).
* **DO** use `think` extensively to show your work.
* **Your final step MUST be to provide your answer in the `_AgentTestGenerationOutput` format.**
"""

async def _invoke_agentic_llm(
    raw_request: str,
    reflections_text: str,
    prev_dataset_examples: str, 
    agent_name: str,
    user_id: str,
    available_tool_names: List[str],
    resource_hints: str,
    n_tests: int,
) -> List[DatasetExample]:
    
    logger.info(f"plan.test-llm: Starting agentic test generation for {n_tests} tests...")

    # Tools for the generator agent to *plan*
    agent_tools = [think, web_search] + LANGCHAIN_MCP_TOOLS
    
    system_prompt = AGENTIC_GENERATOR_SYSTEM_PROMPT.format(
        n_tests=n_tests,
        raw_request=raw_request,
        reflections_text=reflections_text,
        prev_dataset_examples=prev_dataset_examples,
        resource_hints=resource_hints,
        available_tool_names="\n".join(available_tool_names),
    )

    # Use the constants LLM
    test_generation_agent = create_agent(
        model=LLM, 
        tools=agent_tools,
        system_prompt=system_prompt,
        state_schema=EvalAgentState, # Re-use existing state
        response_format=_AgentTestGenerationOutput,
    )
    
    logger.info("Invoking agentic test generator...")
    
    try:
        result = await test_generation_agent.ainvoke(
            {"messages": [HumanMessage(content=f"Please generate {n_tests} complex test cases.")]},
            config=RunnableConfig(recursion_limit=50),
        )
        
        output: _AgentTestGenerationOutput = result.get("structured_response")
        
        if not output or not output.dataset_examples:
            raise ValueError("Agentic test generator failed to return valid output.")

        logger.info(f"Agentic generator summary: {output.generation_summary}")
        
        # Assign IDs
        for example in output.dataset_examples:
            if not example.example_id:
                example.example_id = str(uuid4())
            example.status = "active"
            
        return output.dataset_examples[:n_tests]
        
    except Exception as e:
        logger.error(f"Agentic test generator failed: {e}. Falling back to genetic method.")
        # Fallback to one empty new test to avoid crashing
        return await _invoke_genetic_llm(
            raw_request, reflections_text, prev_dataset_examples, 
            agent_name, user_id, available_tool_names, resource_hints, 
            force_new_test=True # Force one new test
        )

# --- END ADDED SECTION ---


async def generate_eval_plan(state: EvalAgentState) -> dict:
    agent_name = state.github_context.agent_name

    if not state.user_context or not state.user_context.user_id:
        raise ValueError("UserContext with user_id is required to plan")
    user_id = state.user_context.user_id
 
    # Get top 3 most relevant reflections + their evidence using GraphRAG
    reflections_text = await graph_rag_retrieval(
        query="what previous tests failed and why?",
        agent_name=agent_name,
        user_id=user_id,
        limit=3
    )

    # Get just the inputs from the most recent run
    previous_inputs = [res.dataset_example.input_message for res in state.latest_results]

    tool_entries: Dict[str, ToolEntry] = {}
    available_tools: List[str] = []

    # We'll initialize context_for_scoring here to ensure it's always defined
    context_for_scoring = ""

    if state.mcp_services:
        try:
            tool_entries = await load_tool_entries(state.mcp_services)
            context_for_scoring = "\n".join( 
                filter(None, [state.user_context.raw_request, reflections_text])
            )
            prioritized = await select_relevant_tools(
                tool_entries,
                context_for_scoring,
                max_total=20,
                max_per_service=5,
            )
            if not prioritized:
                prioritized = sorted({entry.name for entry in tool_entries.values()})
            available_tools = prioritized
            logger.info(
                "Found %d prioritized MCP tools for test generation.",
                len(available_tools),
            )
        except Exception as exc:
            logger.error(f"Failed to load MCP tools for test generation: {exc}")
    else:
        logger.info("No MCP services configured; tool prompts will be limited to system tools.")

    resource_hints = format_resource_hints(state.mcp_resources)

    log_entry = ToolSelectionLog(
        selection_context=context_for_scoring,
        selected_tools=available_tools
    )

    # --- MODIFIED: Add feature flag for generation strategy ---
    
    # Default to 'false' (the new agentic way)
    USE_GENETIC_TEST_GENERATION = os.getenv("USE_GENETIC_TEST_GENERATION", "false").lower() == "true"
    
    if USE_GENETIC_TEST_GENERATION:
        logger.info("Using 'genetic' (structured prompt) test generation.")
        dataset_examples = await _invoke_genetic_llm(
            raw_request=state.user_context.raw_request,
            reflections_text=reflections_text,
            prev_dataset_examples=json.dumps(previous_inputs, indent=2),
            agent_name=agent_name,
            user_id=user_id,
            available_tools=available_tools,
            resource_hints=resource_hints,
        )
    else:
        logger.info("Using 'agentic' (create_agent) test generation.")
        dataset_examples = await _invoke_agentic_llm(
            raw_request=state.user_context.raw_request,
            reflections_text=reflections_text,
            prev_dataset_examples=json.dumps(previous_inputs, indent=2),
            agent_name=agent_name,
            user_id=user_id,
            available_tool_names=available_tools, # Pass tool *names* as guide
            resource_hints=resource_hints,
            n_tests=N_TEST_CASES,
        )
    
    # --- END MODIFIED SECTION ---

    _validate_generated_actions(dataset_examples, tool_entries)

    logger.info("plan.generate: produced %d tests (agent=%s)", len(dataset_examples), agent_name)
    return {
        "dataset_examples": dataset_examples,
        "tool_selection_log": log_entry,
    }
