"""
This file contains code for the plan node of the eval agent. 
This is also responsible for generating the test cases for the target agent.
"""
import os
import json
import re
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
from shared.schema import GithubContext, UserContext, SandboxContext
from shared.logger import get_logger

logger = get_logger("eval_agent.plan")


CANDIDATE_GENERATION_PROMPT = """### PROMPT: CANDIDATE_GENERATION_PROMPT (EVAL_AGENT) ###
You are a creative, adversarial QA analyst. Your goal is to brainstorm a wide variety of test cases based on fundamental software engineering principles.

<USER_EXPECTATION>
{user_expectation}
</USER_EXPECTATION>

<PAST_EVAL_REFLECTIONS>
{reflections_text}
</PAST_EVAL_REFLECTIONS>

<RECENTLY_RUN_TESTS>
{prev_dataset_examples}
</RECENTLY_RUN_TESTS>

<TEST_CATEGORIES_TO_GENERATE>
You should generate test cases that cover the following fundamental categories:
1.  **Basic Logic:** `if/else`, `elif`, nested conditions, boolean logic (`and`, `or`, `not`).
2.  **Loops:** `for` loops, `while` loops, loops on empty lists, loops on single-item lists, `break`, `continue`.
3.  **Data Structures:**
    * **Lists:** Empty lists, single-item lists, lists with duplicates, nested lists.
    * **Dictionaries:** Empty dicts, dicts with missing keys, dicts with `None` values, nested dicts.
    * **Strings:** Empty strings, strings with special characters, multi-line strings.
    * **Other:** `None` inputs, `sets`, `tuples`.
4.  **Error Conditions:** `try/except` blocks, `raise` statements, divide-by-zero, index-out-of-bounds, `KeyError`.
5.  **OOP (if applicable):** Class instantiation, method calls, inheritance, class variables.
6.  **Adversarial Inputs:** Inputs designed to break simple parsers, inputs that *almost* match but are wrong, inputs that test for flakiness (e.g., non-determinism).
</TEST_CATEGORIES_TO_GENERATE>

<CONSTRAINTS>
- Generate {N_CANDIDATES} diverse test case candidates, drawing inspiration from the <TEST_CATEGORIES_TO_GENERATE>.
- Use <PAST_EVAL_REFLECTIONS> to create *harder, novel* tests that explore the *root cause* of past failures.
- Do not repeat any input_message from <RECENTLY_RUN_TESTS>.
- **This is a brainstorm, so prioritize creativity and breadth over perfect refinement.**
</CONSTRAINTS>
"""

FINAL_SELECTION_PROMPT = """### PROMPT: FINAL_SELECTION_PROMPT (EVAL_AGENT) ###
You are a senior, highly critical QA Lead. Your job is to review a list of test case candidates and select *only* the most effective ones.

<USER_EXPECTATION>
{user_expectation}
</USER_EXPECTATION>

<PAST_EVAL_REFLECTIONS>
{reflections_text}
</PAST_EVAL_REFLECTIONS>

<CANDIDATE_TEST_CASES>
Here are {N_CANDIDATES} candidates brainstormed by your team:
{candidate_examples_json}
</CANDIDATE_TEST_CASES>

<CONSTRAINTS>
- Your goal is to find *novel* bugs.
- **You MUST select exactly {N_TEST_CASES} test cases.**
- **Critically analyze the candidates:**
    - Discard any that are too simple or duplicates.
    - Discard any that repeat failure modes we've already seen in the reflections, unless they test it in a new, harder way.
    - **Prioritize tests that are complex, target known weak spots (from reflections), or test novel edge cases.**
- **Learn from the "Past Test Critique" in the reflections.** If a critique said "tests were too simple," you MUST select the hardest ones.
- Return *only* your final selection of {N_TEST_CASES} test cases.
</CONSTRAINTS>
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
    prev_dataset_examples: str,
) -> List[DatasetExample]:
    
    class _TestGenerationOutput(BaseModel):
        """
        Explicitly defined internal class for the test generation LLM output. 
        Since structured output does not support lists, we need to define it explicitly.
        """
        dataset_examples: List[DatasetExample]

    # Use a smart, critical LLM for this reasoning chain
    _smart_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
    test_generation_llm = _smart_llm.with_structured_output(_TestGenerationOutput)

    # --- Step 1: Generate a large pool of candidates ---
    
    # Generate 3x the number of tests we actually need
    N_CANDIDATES = N_TEST_CASES * 3 
    
    logger.info(
        "plan.test-llm: Step 1 - Generating %d candidates...", N_CANDIDATES
    )
    
    candidate_prompt = CANDIDATE_GENERATION_PROMPT.format(
        N_CANDIDATES=N_CANDIDATES,
        user_expectation=user_expectation,
        reflections_text=reflections_text,
        prev_dataset_examples=prev_dataset_examples,
    )
    
    try:
        candidate_output: _TestGenerationOutput = await test_generation_llm.ainvoke(candidate_prompt)
        candidates = candidate_output.dataset_examples
        logger.info(
            "plan.test-llm: Step 1 - Successfully generated %d candidates.", len(candidates)
        )
    except Exception as e:
        logger.error(f"plan.test-llm: Step 1 - FAILED to generate candidates. {e}")
        # Fallback: If candidate generation fails, return an empty list.
        return []

    if not candidates:
        logger.warning("plan.test-llm: Step 1 - No candidates were generated.")
        return []

    # --- Step 2: Critique and Select the best N_TEST_CASES ---

    logger.info(
        "plan.test-llm: Step 2 - Critiquing and selecting %d final tests...", N_TEST_CASES
    )
    
    candidate_examples_json = json.dumps(
        [ex.model_dump() for ex in candidates], indent=2
    )

    selection_prompt = FINAL_SELECTION_PROMPT.format(
        N_TEST_CASES=N_TEST_CASES,
        N_CANDIDATES=len(candidates),
        user_expectation=user_expectation,
        reflections_text=reflections_text,
        candidate_examples_json=candidate_examples_json,
    )

    try:
        final_selection_output: _TestGenerationOutput = await test_generation_llm.ainvoke(selection_prompt)
        final_tests = final_selection_output.dataset_examples
        logger.info(
            "plan.test-llm: Step 2 - Successfully selected %d final tests.", len(final_tests)
        )
    except Exception as e:
        logger.error(f"plan.test-llm: Step 2 - FAILED to select final tests. {e}")
        # Fallback: If selection fails, return the first N tests from candidates
        final_tests = candidates[:N_TEST_CASES]
        
    if not final_tests:
        logger.warning("plan.test-llm: Step 2 - No final tests were selected.")
        return []

    # Assign final Example IDs
    for example in final_tests:
        example.example_id = str(uuid4())

    logger.info(
        "plan.test-llm: Chain of Thought generation complete. Produced %d tests.",
        len(final_tests),
    )
    
    # Return *only* the final, selected tests
    return final_tests


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
    )

    logger.info("plan.generate: produced %d tests (agent=%s)", len(dataset_examples), agent_name)
    return {
        "dataset_examples": dataset_examples,
        "reflections_used_for_planning": reflections_text,
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
