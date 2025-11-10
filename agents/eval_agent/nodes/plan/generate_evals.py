import asyncio
import json
from typing import List
from uuid import uuid4
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.constants import NEO4J_GRAPH, N_TEST_CASES
from agents.eval_agent.reflection_store import graph_rag_retrieval
from shared.schema import DatasetExample
from shared.logger import get_logger
logger = get_logger("eval_agent.plan")


class _TestGenerationOutput(BaseModel):
    """Helper for structured output"""
    dataset_example: DatasetExample


COMMON_INSTRUCIONS = """\n\n
4.  **Generate Test Package:**
    * **Write Buggy Code:** Create a new, simple Python code snippet with your selected mutation (e.g., a function that will divide by zero).
    * **Write Visible Tests:** Write 1-2 *passing* `unittest` cases that a *correct* solution should pass. These *must not* trigger the bug.
    * **Write Hidden Tests:** Write 1-2 `unittest` cases that *specifically* test the bug.
      * **Your test code MUST import the functions from `solution.py` (e.g., `from solution import process_data`).**
      * **CRITICAL:** When testing for errors (like division by zero, index errors, etc.), write *robust* tests.
      * **Bad Test:** `with self.assertRaises(ZeroDivisionError):` (This is too specific!)
      * **Good Test:** `with self.assertRaises((ZeroDivisionError, ValueError, TypeError, Exception)):` (This is better, as it accepts any valid error-handling strategy the agent might use).
      * **Best Test:** If possible, write a test that checks the *behavior*. For example, wrap the call in a `try/except` block and assert that an exception *was* raised, without being too specific about its type.
5.  **Format Output (CRITICAL):**
    <reasoning>
        # why this is a good test
        "A test for..."
    </reasoning>

    <input_message>
        <buggy_code>
            # include only the buggy code here
            ```python
            ...
            ```
        </buggy_code>
        <visible_tests>
            # include only 1-2 visible tests here
            from solution import <function_to_test>
            import unittest
            ....
        </visible_tests>
        <instructions>
            # what should the target agent do?
            <example>
                please fix the buggy code above so that all visible tests pass.
                you don't need to generate hidden test cases. we'll handle that part.
            </example>
        </instructions>
    </input_message>

    <expected_output>
        <candidate_solution>
            # include a possible fixed version of the buggy code here
            ```python
            ...
            ```
        </candidate_solution>
        <hidden_tests>
            # include only 2-4 hidden tests here
            from solution import <function_to_test>
            import unittest
            ....
        </hidden_tests>
    </expected_output>

    <test_generation_instructions>
    1.  **Test Imports:** Your hidden tests (`expected_output`) *must* import from `solution.py` (e.g., `from solution import outer`).
    2.  **Testing Globals:** When testing global variables, you **MUST** import the module itself (e.g., `import solution`) and access the variable via the module (e.g., `self.assertEqual(solution.global_var, 1)`). **DO NOT** use `from solution import global_var`.
    3.  **Test Isolation:** Use a `setUp(self)` method to reset any global state before each test. (e.g., `def setUp(self): import solution; solution.global_var = 0`).
    </test_generation_instructions>
    
Provide your final output as *only* the new `DatasetExample` Pydantic object.
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
1.  **Analyze Parent:** What simple case does the parent test? (e.g., 'basic dict access', 'simple string replacement')
2.  **Brainstorm Mutations:** Look at the "Genetic Operators" below. Brainstorm 3 *different* ways to mutate the "Parent Test" to make it harder and more complex.
    * **Genetic Operators:** `add_error_condition` (force a `KeyError`, `IndexError`, `ZeroDivisionError`), `add_adversarial_input` (non-UTF-8, empty strings, `None` values, large inputs), `nest` (add nested data structures), `repeat` (force it to run in a loop).
3.  **Select Best Mutation:** Which of your 3 ideas is *most likely* to cause a sophisticated agent to fail?
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
You MUST use the following Chain of Thought:

**Chain of Thought (MANDATORY):**
1.  **Analyze Weaknesses:** Based on "Past Reflections," what is the agent's *biggest* known weakness? (If no reflections, focus on the "Raw request").
2.  **Brainstorm New Attack:** Brainstorm 3 *new, different* test scenarios based on this weakness.
    * **Adversarial Techniques:** `add_error_condition` (`KeyError`, `IndexError`, `ZeroDivisionError`, `TypeError`), `add_adversarial_input` (non-UTF-8, empty strings, `None` values, edge-case strings, large inputs), `nest` (use deeply nested data), `repeat` (test with loops).
3.  **Select Best Attack:** Which of your 3 ideas is *most novel* and *not* in "Recently Run Tests"?
""" + COMMON_INSTRUCIONS


async def _invoke_test_generation_llm(
    raw_request: str,
    reflections_text: str,
    prev_dataset_examples: str, # JSON string of recent inputs
    agent_name: str,
    user_id: str,
) -> List[DatasetExample]:
    
    logger.info("plan.test-llm: Starting evolutionary test generation...")
    
    # Use a smart, critical LLM for this
    _smart_llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)
    test_generation_llm = _smart_llm.with_structured_output(_TestGenerationOutput)

    new_generation: List[DatasetExample] = []
    
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
        return [dict(r) for r in result]

    async def run_crossover(parents: List[dict]):
        """Run the Crossover prompt"""
        logger.info("plan.test-llm: Running Crossover...")
        prompt = CROSSOVER_PROMPT.format(
            parent_1_json=json.dumps(parents[0], indent=2),
            parent_2_json=json.dumps(parents[1], indent=2),
            reflections_text=reflections_text
        )
        output = await test_generation_llm.ainvoke(prompt)
        new_generation.append(output.dataset_example)

    async def run_mutation(parent: dict):
        """Run the Mutation prompt"""
        logger.info("plan.test-llm: Running Mutation...")
        prompt = MUTATION_PROMPT.format(
            parent_test_json=json.dumps(parent, indent=2),
            reflections_text=reflections_text
        )
        output = await test_generation_llm.ainvoke(prompt)
        new_generation.append(output.dataset_example)

    async def run_new_test():
        """Run the New Test prompt"""
        logger.info("plan.test-llm: Running New Test generation...")
        prompt = NEW_TEST_PROMPT.format(
            raw_request=raw_request,
            reflections_text=reflections_text,
            prev_dataset_examples=prev_dataset_examples
        )
        output = await test_generation_llm.ainvoke(prompt)
        new_generation.append(output.dataset_example)

    # --- Genetic Algorithm "Breeding" Strategy ---
    # Example for N_TEST_CASES = 3: 1 Crossover, 1 Mutation, 1 New
    # This ensures a mix of exploitation (Crossover/Mutation) and exploration (New)
    
    # 1. Crossover (Exploitation)
    if len(new_generation) < N_TEST_CASES:
        fit_parents = await get_parent(passed=False, k=2) # Get 2 FAILED tests
        if len(fit_parents) == 2:
            await run_crossover(fit_parents)
        else:
            logger.warning("Could not find 2 'fit' (failed) parents for crossover. Skipping.")

    # 2. Mutation (Exploitation)
    if len(new_generation) < N_TEST_CASES:
        unfit_parent = await get_parent(passed=True, k=1) # Get 1 PASSED test
        if unfit_parent:
            await run_mutation(unfit_parent[0])
        else:
            logger.warning("Could not find 1 'unfit' (passed) parent for mutation. Skipping.")

    # 3. New Tests (Exploration)
    # Fill the remaining slots with brand new tests
    while len(new_generation) < N_TEST_CASES:
        await run_new_test()

    # --- Assign final Example IDs ---
    for example in new_generation:
        example.example_id = str(uuid4())

    logger.info(
        "plan.test-llm: Evolutionary generation complete. Produced %d tests.",
        len(new_generation),
    )
    
    return new_generation[:N_TEST_CASES] # Ensure we only return the number requested




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

    dataset_examples = await _invoke_test_generation_llm(
        raw_request=state.user_context.raw_request,
        reflections_text=reflections_text,
        prev_dataset_examples=json.dumps(previous_inputs, indent=2),
        agent_name=agent_name,
        user_id=user_id,
    )

    logger.info("plan.generate: produced %d tests (agent=%s)", len(dataset_examples), agent_name)
    return {
        "dataset_examples": dataset_examples,
    }