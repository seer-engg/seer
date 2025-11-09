"""
This file contains code for the plan node of the eval agent. 
This is also responsible for generating the test cases for the target agent.
"""
import os
import json
import re
import asyncio
from typing import List, Tuple, Optional
from uuid import uuid4

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.constants import LLM, N_TEST_CASES
from agents.eval_agent.models import (
    EvalAgentState,
    DatasetExample,
)
from agents.eval_agent.reflection_store import graph_rag_retrieval
from sandbox import (
    TARGET_AGENT_COMMAND,
    TARGET_AGENT_PORT,
    deploy_server_and_confirm_ready,
    initialize_e2b_sandbox,
    setup_project,
)
from agents.eval_agent.constants import NEO4J_GRAPH
from shared.schema import GithubContext, UserContext, SandboxContext
from shared.logger import get_logger

logger = get_logger("eval_agent.plan")

class _TestGenerationOutput(BaseModel):
    """Helper for structured output"""
    dataset_example: DatasetExample


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
4.  **Generate Test Package:**
    * **Write Buggy Code:** Create a new, simple Python code snippet with your selected mutation (e.g., a function that will divide by zero).
    * **Write Visible Tests:** Write 1-2 *passing* `unittest` cases that a *correct* solution should pass. These *must not* trigger the bug.
    * **Write Hidden Tests:** Write 1-2 `unittest` cases that *specifically* test the bug.
      * **Your test code MUST import the functions from `solution.py` (e.g., `from solution import process_data`).**
      * **CRITICAL:** When testing for errors (like division by zero, index errors, etc.), write *robust* tests.
      * **Bad Test:** `with self.assertRaises(ZeroDivisionError):` (This is too specific!)
      * **Good Test:** `with self.assertRaises((ZeroDivisionError, ValueError, TypeError, Exception)):` (This is better, as it accepts any valid error-handling strategy the agent might use).
      * **Best Test:** If possible, write a test that checks the *behavior*. For example, wrap the call in a `try/except` block and assert that an exception *was* raised, without being too specific about its type.
5.  **Format Output:**
    * `reasoning`: "A mutation of '{{parent_test_reason}}' to also test {{new_bug_description}}."
    * `input_message`: A string containing the buggy code (in ` ```python `) and the *visible tests* (in ` ```python `).
    * `expected_output`: A string containing only the raw Python code for the hidden tests. Do NOT wrap this in ```python markdown fences.

Provide your final output as *only* the new `DatasetExample` Pydantic object.
"""

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
4.  **Generate Test Package:**
    * **Write Buggy Code:** Write the new hybrid buggy code.
    * **Write Visible Tests:** Write 1-2 *passing* `unittest` cases for the "happy path."
    * **Write Hidden Tests:** Write 1-2 `unittest` cases that *specifically* test the bug.
      * **Your test code MUST import the functions from `solution.py` (e.g., `from solution import process_data`).**
      * **CRITICAL:** When testing for errors (like division by zero, index errors, etc.), write *robust* tests.
      * **Bad Test:** `with self.assertRaises(ZeroDivisionError):` (This is too specific!)
      * **Good Test:** `with self.assertRaises((ZeroDivisionError, ValueError, TypeError, Exception)):` (This is better, as it accepts any valid error-handling strategy the agent might use).
      * **Best Test:** If possible, write a test that checks the *behavior*. For example, wrap the call in a `try/except` block and assert that an exception *was* raised, without being too specific about its type.
5.  **Format Output:**
    * `reasoning`: "A hybrid test combining {{parent_1_failure}} and {{parent_2_failure}}."
    * `input_message`: A string containing the hybrid buggy code and the *visible tests*.
    * `expected_output`: A string containing only the raw Python code for the hidden tests. Do NOT wrap this in ```python markdown fences.

Provide your final output as *only* the new `DatasetExample` Pydantic object.
"""

NEW_TEST_PROMPT = """### PROMPT: NEW_TEST_GENERATOR (EVAL_AGENT) ###
You are an adversarial QA analyst. Your goal is to brainstorm *one* novel test case.

**User Expectation:**
{user_expectation}

**Past Reflections (Agent Weaknesses):**
{reflections_text}

**Recently Run Tests (Do Not Repeat These):**
{prev_dataset_examples}

**Your Task:**
Create one new, creative, and *hard* `DatasetExample`.
You MUST use the following Chain of Thought:

**Chain of Thought (MANDATORY):**
1.  **Analyze Weaknesses:** Based on "Past Reflections," what is the agent's *biggest* known weakness? (If no reflections, focus on the "User Expectation").
2.  **Brainstorm New Attack:** Brainstorm 3 *new, different* test scenarios based on this weakness.
    * **Adversarial Techniques:** `add_error_condition` (`KeyError`, `IndexError`, `ZeroDivisionError`, `TypeError`), `add_adversarial_input` (non-UTF-8, empty strings, `None` values, edge-case strings, large inputs), `nest` (use deeply nested data), `repeat` (test with loops).
3.  **Select Best Attack:** Which of your 3 ideas is *most novel* and *not* in "Recently Run Tests"?
4.  **Generate Test Package:**
    * **Write Buggy Code:** Create a simple Python code snippet that has your selected bug (e.g., `def process(data): return data['key']`).
    * **Write Visible Tests:** Write 1-2 *passing* `unittest` cases (e.g., `test_happy_path(self): self.assertEqual(process({{'key': 'val'}}), 'val')`).
    * **Write Hidden Tests:** Write 1-2 `unittest` cases that *specifically* test the bug.
      * **Your test code MUST import the functions from `solution.py` (e.g., `from solution import process_data`).**
      * **CRITICAL:** When testing for errors (like division by zero, index errors, etc.), write *robust* tests.
      * **Bad Test:** `with self.assertRaises(ZeroDivisionError):` (This is too specific!)
      * **Good Test:** `with self.assertRaises((ZeroDivisionError, ValueError, TypeError, Exception)):` (This is better, as it accepts any valid error-handling strategy the agent might use).
      * **Best Test:** If possible, write a test that checks the *behavior*. For example, wrap the call in a `try/except` block and assert that an exception *was* raised, without being too specific about its type.
5.  **Format Output:**
    * `reasoning`: "A new test for {{description_of_bug}}."
    * `input_message`: A string containing the buggy code (in ` ```python `) and the *visible tests* (in ` ```python `).
    * `expected_output`: A string containing only the raw Python code for the hidden tests. Do NOT wrap this in ```python markdown fences.

Provide your final output as *only* the new `DatasetExample` Pydantic object.
"""


def _parse_github_url(url: str, branch_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Parse a GitHub URL and extract the repository URL and branch name.
    
    Handles both:
    - Web URLs: https://github.com/owner/repo/tree/branch-name
    - Git URLs: https://github.com/owner/repo or https://github.com/owner/repo.git
    
    Args:
        url: The GitHub URL (can be a web URL with /tree/ or a git URL)
        branch_name: Optional branch name to use if not in URL
    
    Returns:
        Tuple of (repo_url, branch_name)
    """
    # Pattern to match GitHub web URLs with /tree/ path
    # The branch name can contain slashes, so we match everything after /tree/ 
    # up to an optional trailing slash or path
    web_url_pattern = r'^(https?://github\.com/[^/]+/[^/]+)/tree/([^/]+(?:/[^/]+)*)/?(?:/.+)?$'
    match = re.match(web_url_pattern, url)
    
    if match:
        # Extract repo URL and branch from web URL
        repo_url = match.group(1)
        extracted_branch = match.group(2)
        logger.info(f"Parsed GitHub web URL: repo_url={repo_url}, branch={extracted_branch}")
        return repo_url, extracted_branch
    
    # If it's a standard git URL, use it as-is
    # Remove trailing .git if present for consistency
    repo_url = re.sub(r'\.git$', '', url)
    final_branch = branch_name or "main"
    
    return repo_url, final_branch


async def _invoke_test_generation_llm(
    user_expectation: str,
    reflections_text: str,
    prev_dataset_examples: str, # JSON string of recent inputs
    agent_name: str,
    user_id: str,
) -> List[DatasetExample]:
    
    logger.info("plan.test-llm: Starting evolutionary test generation...")
    
    # Use a smart, critical LLM for this
    _smart_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
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
            user_expectation=user_expectation,
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


async def _ensure_target_agent_config(state: EvalAgentState) -> dict:
    last_human = None
    for msg in reversed(state.messages or []):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            last_human = msg
            break
    if last_human is None:
        raise ValueError("No human message to extract from")

    instruction = (
        "Extract the following fields from the user's latest message about the target agent:\n"
        "- github_context: the GitHub context for the target agent\n"
        "- user_context: the user context for the target agent\n"
    )

    class TargetAgentExtractionContext(BaseModel):
        """Context for extracting the target agent's GitHub and user context."""
        github_context: GithubContext
        user_context: UserContext

    extractor = LLM.with_structured_output(TargetAgentExtractionContext)
    context: TargetAgentExtractionContext = await extractor.ainvoke(f"{instruction}\n\nUSER:\n{last_human.content}")
    context.user_context.user_raw_request = last_human.content
    
    # Normalize the GitHub URL in case it's a web URL with /tree/ in it
    normalized_repo_url, normalized_branch = _parse_github_url(
        context.github_context.repo_url, 
        context.github_context.branch_name
    )
    context.github_context.repo_url = normalized_repo_url
    context.github_context.branch_name = normalized_branch
    
    return {
        "github_context": context.github_context,
        "user_context": context.user_context,
    }


async def _provision_target_agent(state: EvalAgentState) -> dict:
    repo_url = state.github_context.repo_url
    branch_name = state.github_context.branch_name

    if not state.sandbox_context:
        github_token = os.getenv("GITHUB_TOKEN")
        logger.info(
            "plan.provision: provisioning sandbox (repo=%s branch=%s)",
            repo_url,
            branch_name,
        )

        sbx, repo_dir, resolved_branch = await initialize_e2b_sandbox(
            repo_url=repo_url,
            branch_name=branch_name,
            github_token=github_token,
        )
        sandbox_branch = resolved_branch or branch_name
        sandbox_id = sbx.sandbox_id

        await setup_project(sandbox_id, repo_dir, "pip install -e .")

        sandbox, _ = await deploy_server_and_confirm_ready(
            cmd=TARGET_AGENT_COMMAND,
            sb=sbx,
            cwd=repo_dir,
        )

        deployment_url = sandbox.get_host(TARGET_AGENT_PORT)
        if not deployment_url.startswith("http"):
            deployment_url = f"https://{deployment_url}"

        logger.info("plan.provision: sandbox ready at %s", deployment_url)

        return {
            "sandbox_context": SandboxContext(
                sandbox_id=sandbox_id,
                working_branch=sandbox_branch,
            ),
        }
    else:
        logger.info("plan.provision: reusing deployment url %s", state.sandbox_context.deployment_url)
        return {}


async def _generate_eval_plan(state: EvalAgentState) -> dict:
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
        user_expectation=state.user_context.user_expectation,
        reflections_text=reflections_text,
        prev_dataset_examples=json.dumps(previous_inputs, indent=2),
        agent_name=agent_name,
        user_id=user_id,
    )

    logger.info("plan.generate: produced %d tests (agent=%s)", len(dataset_examples), agent_name)
    return {
        "dataset_examples": dataset_examples,
    }


def build_plan_subgraph():
    """Build the plan subgraph."""
    builder = StateGraph(EvalAgentState)
    builder.add_node("ensure-config", _ensure_target_agent_config)
    builder.add_node("provision-target", _provision_target_agent)
    builder.add_node("generate-tests", _generate_eval_plan)

    builder.add_edge(START, "ensure-config")
    builder.add_edge("ensure-config", "provision-target")
    builder.add_edge("provision-target", "generate-tests")
    builder.add_edge("generate-tests", END)

    return builder.compile()
