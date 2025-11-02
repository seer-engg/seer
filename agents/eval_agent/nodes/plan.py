import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.deps import LLM, logger
from agents.eval_agent.models import (
    AgentSpec,
    DeploymentContext,
    EvalAgentState,
    GeneratedTestCase,
    GeneratedTests,
    TargetAgentConfig,
)
from agents.eval_agent.prompts import (
    EVAL_AGENT_SPEC_PROMPT,
    EVAL_AGENT_TEST_GEN_PROMPT,
)
from agents.eval_agent.reflection_store import (
    format_previous_inputs,
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


async def _invoke_agent_spec_llm(
    cfg: TargetAgentConfig,
    repo_url_hint: str,
    branch_hint: str,
    deployment_url_hint: str,
) -> tuple[AgentSpec, Dict[str, Any]]:
    spec_prompt = EVAL_AGENT_SPEC_PROMPT.format(
        expectations=cfg.expectations,
        agent_name=cfg.graph_name,
        agent_repo=repo_url_hint,
        agent_branch=branch_hint,
        deployment_url=deployment_url_hint,
    )
    spec_llm = LLM.with_structured_output(AgentSpec)
    spec_obj: AgentSpec = await spec_llm.ainvoke(spec_prompt)

    trace = {
        "prompt": spec_prompt,
        "response": spec_obj.model_dump(mode="json"),
    }
    logger.info(
        "plan.spec-llm: generated spec for agent=%s (prompt_chars=%d)",
        cfg.graph_name,
        len(spec_prompt),
    )
    return spec_obj, trace


async def _invoke_test_generation_llm(
    spec_obj: AgentSpec,
    reflections_text: str,
    prev_inputs_text: str,
) -> tuple[GeneratedTests, Dict[str, Any]]:
    spec_json = json.dumps(spec_obj.model_dump(mode="json"), indent=2)
    augmented_prompt = EVAL_AGENT_TEST_GEN_PROMPT.format(
        spec_json=spec_json,
        reflections_text=reflections_text,
        prev_inputs_text=prev_inputs_text,
    )
    generated_runnable = LLM.with_structured_output(GeneratedTests)
    generated: GeneratedTests = await generated_runnable.ainvoke(augmented_prompt)

    trace = {
        "prompt": augmented_prompt,
        "response": [tc.model_dump(mode="json") for tc in generated.test_cases],
    }
    logger.info(
        "plan.test-llm: generated %d tests (prompt_chars=%d)",
        len(generated.test_cases),
        len(augmented_prompt),
    )
    return generated, trace


async def _ensure_target_agent_config(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is not None:
        updates: Dict[str, Any] = {}
        if state.pending_followup and state.codex_followup_branch:
            new_branch = state.codex_followup_branch.strip()

            cfg_updates: Dict[str, Any] = {"url": None}
            if new_branch and cfg.branch_name != new_branch:
                cfg_updates["branch_name"] = new_branch
                logger.info(
                    "plan.ensure-config: switched target branch to Codex follow-up branch %s",
                    new_branch,
                )
            cfg = cfg.model_copy(update=cfg_updates)

            deployment = state.deployment or DeploymentContext()
            metadata = dict(deployment.metadata)

            # Drop stale deployment hints so provisioning rebuilds on the Codex branch
            for stale_key in (
                "deployment_url",
                "codex_handoff_status",
                "last_boot_at",
                "last_upload_at",
            ):
                metadata.pop(stale_key, None)
            if new_branch:
                metadata["branch_name"] = new_branch

            deployment_updates: Dict[str, Any] = {
                "url": None,
                "branch_name": new_branch or deployment.branch_name,
                "repo_url": cfg.repo_url or deployment.repo_url,
                "sandbox_id": None,
                "sandbox_repo_dir": None,
                "sandbox_branch": None,
                "metadata": metadata,
            }

            updates.update(
                {
                    "pending_followup": False,
                    "target_agent_config": cfg,
                    "deployment": deployment.model_copy(update=deployment_updates),
                }
            )
            return updates

        if cfg.repo_url:
            return {"target_agent_config": cfg}

    last_human = None
    for msg in reversed(state.messages or []):
        try:
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                last_human = msg
                break
        except Exception:
            continue
    if last_human is None:
        raise ValueError("Missing target_agent_config and no human message to extract from")

    instruction = (
        "Extract the following fields from the user's latest message about the target agent:\n"
        "- graph_name: the LangGraph graph name to evaluate (NOT the assistant/thread id).\n"
        "- repo_url: canonical Git URL (https preferred) for the agent's source repository.\n"
        "- expectations: verbatim expectations text describing desired behavior.\n"
        "- url: fully-qualified deployment URL if the agent is already running; leave empty/null otherwise.\n"
        "- branch_name: Git branch to evaluate (default to 'main' if unspecified).\n"
        "- setup_script: single-line shell command to prepare the project inside the sandbox before launch (default 'pip install -e .')."
    )

    extractor = LLM.with_structured_output(TargetAgentConfig)
    cfg = await extractor.ainvoke(f"{instruction}\n\nUSER:\n{last_human.content}")
    return {"target_agent_config": cfg}


async def _provision_target_agent(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("_provision_target_agent requires target_agent_config to be set")

    repo_url = (cfg.repo_url or "").strip()
    branch_name = (cfg.branch_name or "main").strip() or "main"
    setup_script = (cfg.setup_script or "pip install -e .").strip() or "pip install -e ."

    deployment = state.deployment or DeploymentContext()
    deployment_url = cfg.url or deployment.url
    sandbox_id = deployment.sandbox_id
    sandbox_repo_dir = deployment.sandbox_repo_dir
    sandbox_branch = deployment.sandbox_branch or branch_name
    metadata = dict(deployment.metadata)

    if not deployment_url:
        if not repo_url:
            raise ValueError("TargetAgentConfig.repo_url is required when no deployment URL is provided")

        github_token = os.getenv("GITHUB_TOKEN")
        logger.info(
            "plan.provision: provisioning sandbox (repo=%s branch=%s existing_sandbox=%s)",
            repo_url,
            branch_name,
            sandbox_id or "<new>",
        )

        sbx, repo_dir, resolved_branch = await initialize_e2b_sandbox(
            repo_url=repo_url,
            branch_name=branch_name,
            github_token=github_token,
        )
        sandbox_repo_dir = repo_dir
        sandbox_branch = resolved_branch or branch_name
        sandbox_id = sbx.sandbox_id

        await setup_project(sandbox_id, repo_dir, setup_script)

        sandbox, _ = await deploy_server_and_confirm_ready(
            cmd=TARGET_AGENT_COMMAND,
            sb=sbx,
            cwd=repo_dir,
        )

        deployment_url = sandbox.get_host(TARGET_AGENT_PORT)
        if not deployment_url.startswith("http"):
            deployment_url = f"https://{deployment_url}"

        metadata.update(
            {
                "sandbox_id": sandbox_id,
                "sandbox_repo_dir": repo_dir,
                "sandbox_branch": sandbox_branch,
                "setup_script": setup_script,
                "last_boot_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        logger.info("plan.provision: sandbox ready at %s", deployment_url)
    else:
        metadata.setdefault("last_seen_at", datetime.now(timezone.utc).isoformat())
        logger.info("plan.provision: reusing deployment url %s", deployment_url)

    cfg = cfg.model_copy(update={"url": deployment_url, "branch_name": sandbox_branch, "setup_script": setup_script})

    metadata.update(
        {
            "deployment_url": deployment_url,
            "repo_url": repo_url,
            "branch_name": sandbox_branch,
        }
    )

    deployment_updates = {
        "url": deployment_url,
        "repo_url": repo_url or deployment.repo_url,
        "branch_name": sandbox_branch,
        "sandbox_id": sandbox_id,
        "sandbox_repo_dir": sandbox_repo_dir,
        "sandbox_branch": sandbox_branch,
        "setup_script": setup_script,
        "metadata": metadata,
    }
    deployment = deployment.model_copy(update=deployment_updates)

    return {
        "target_agent_config": cfg,
        "deployment": deployment,
    }


async def _generate_eval_plan(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("_generate_eval_plan requires target_agent_config to be set")

    agent_name = cfg.graph_name
    expectations = cfg.expectations
    repo_url_hint = cfg.repo_url or "(missing repo URL)"
    branch_hint = cfg.branch_name or "main"
    deployment_url_hint = cfg.url or state.deployment.url or "Pending sandbox deployment"

    reflections = await load_recent_reflections(agent_name, expectations, limit=5)
    reflections_text = format_reflections_for_prompt(reflections)

    spec_obj, spec_trace = await _invoke_agent_spec_llm(
        cfg=cfg,
        repo_url_hint=repo_url_hint,
        branch_hint=branch_hint,
        deployment_url_hint=deployment_url_hint,
    )

    prev_inputs = state.previous_inputs or []
    prev_inputs_text = format_previous_inputs(prev_inputs)

    generated, test_trace = await _invoke_test_generation_llm(
        spec_obj=spec_obj,
        reflections_text=reflections_text,
        prev_inputs_text=prev_inputs_text,
    )

    test_cases: list[GeneratedTestCase] = []
    for tc in generated.test_cases:
        test_cases.append(
            GeneratedTestCase(
                input_message=tc.input_message,
                expected_behavior=tc.expected_behavior,
                success_criteria=tc.success_criteria,
                expected_output=getattr(tc, "expected_output", None) or "",
            )
        )

    new_prev = list(prev_inputs)
    new_prev.extend([tc.input_message for tc in test_cases])

    planning_trace = dict(getattr(state, "planning_trace", {}) or {})
    planning_trace.update(
        {
            "spec_generation": spec_trace,
            "test_generation": test_trace,
            "reflections_context": json.loads(reflections_text) if reflections_text not in {"[]", "(none)"} else [],
            "previous_inputs": json.loads(prev_inputs_text) if prev_inputs_text != "[]" else [],
        }
    )

    logger.info("plan.generate: produced %d tests (agent=%s)", len(test_cases), agent_name)
    return {
        "test_cases": test_cases,
        "previous_inputs": new_prev,
        "dataset_name": "",
        "experiment_name": "",
        "planning_trace": planning_trace,
    }


def build_plan_subgraph():
    builder = StateGraph(EvalAgentState)
    builder.add_node("ensure-config", _ensure_target_agent_config)
    builder.add_node("provision-target", _provision_target_agent)
    builder.add_node("generate-tests", _generate_eval_plan)

    builder.add_edge(START, "ensure-config")
    builder.add_edge("ensure-config", "provision-target")
    builder.add_edge("provision-target", "generate-tests")
    builder.add_edge("generate-tests", END)

    return builder.compile()


