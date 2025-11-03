"""
This file contains code for the plan node of the eval agent. 
This is also responsible for generating the test cases for the target agent.
"""
import os
import json
from typing import List
from uuid import uuid4
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.constants import LLM
from agents.eval_agent.models import (
    EvalAgentState,
    DatasetExample,
)
from agents.eval_agent.prompts import (
    EVAL_AGENT_TEST_GEN_PROMPT,
)
from agents.eval_agent.reflection_store import (
    format_reflections_for_prompt,
    load_recent_reflections,
)
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



async def _invoke_test_generation_llm(
    user_expectation: str,
    reflections_text: str,
    prev_dataset_examples: str,
) -> List[DatasetExample]:
    augmented_prompt = EVAL_AGENT_TEST_GEN_PROMPT.format(
        user_expectation=user_expectation,
        reflections_text=reflections_text,
        prev_dataset_examples=prev_dataset_examples,
    )

    class _TestGenerationOutput(BaseModel):
        """
        Explicitly defined internal class for the test generation LLM output. 
        Since structured output does not support lists, we need to define it explicitly.
        """
        dataset_examples: List[DatasetExample]

    test_generation_llm = LLM.with_structured_output(_TestGenerationOutput)
    generated: _TestGenerationOutput = await test_generation_llm.ainvoke(augmented_prompt)
    
    for example in generated.dataset_examples:
        example.example_id = str(uuid4())

    logger.info(
        "plan.test-llm: generated %d tests (prompt_chars=%d)",
        len(generated.dataset_examples),
        len(augmented_prompt),
    )
    return generated.dataset_examples


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
    context = await extractor.ainvoke(f"{instruction}\n\nUSER:\n{last_human.content}")
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
    reflections = await load_recent_reflections(agent_name, 'what previous tests failed',limit=5)
    reflections_text = format_reflections_for_prompt(reflections)

    dataset_examples = await _invoke_test_generation_llm(
        user_expectation=state.user_context.user_expectation,
        reflections_text=reflections_text,
        prev_dataset_examples=json.dumps([example.model_dump() for example in state.dataset_examples], indent=2),
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
