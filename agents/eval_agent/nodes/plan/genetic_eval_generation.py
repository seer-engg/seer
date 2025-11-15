from agents.eval_agent.models import EvalAgentPlannerState
from shared.logger import get_logger
from typing import List
from shared.schema import DatasetExample
from graph_db import NEO4J_GRAPH
import asyncio
import json
from uuid import uuid4
logger = get_logger("eval_agent.plan.genetic_eval_generation")
from langchain_openai import ChatOpenAI
from agents.eval_agent.models import TestGenerationOutput
from shared.resource_utils import format_resource_hints

from agents.eval_agent.constants import (
    N_TEST_CASES,
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





async def genetic_eval_generation(state: EvalAgentPlannerState) -> dict:
    
    logger.info("plan.test-llm: Starting 'genetic' (structured prompt) test generation...")

    available_tools = state.available_tools
    reflections_text = state.reflections_text
    agent_name = state.github_context.agent_name
    user_id = state.user_context.user_id
    raw_request = state.user_context.raw_request
    resource_hints = format_resource_hints(state.mcp_resources)
    previous_inputs = [res.dataset_example.input_message for res in state.latest_results]
    prev_dataset_examples = json.dumps(previous_inputs, indent=2)

    # Use a smart, critical LLM for this
    _smart_llm = ChatOpenAI(
            model="gpt-5-codex",
            use_responses_api=True,            
            output_version="responses/v1",     
            reasoning={"effort": "low"},
        )
    test_generation_llm = _smart_llm.with_structured_output(TestGenerationOutput)

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
    
    return {
        "dataset_examples": new_generation[:N_TEST_CASES],
    }
