from typing import Any, List

from agents.eval_agent.constants import LLM, logger
from agents.eval_agent.models import EvalAgentState, EvalReflection, RunContext
from agents.eval_agent.reflection_store import persist_reflection


def _truncate(text: Any, limit: int = 280) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (truncated {len(text)} chars)"


def reflect_node(state: EvalAgentState) -> dict:
    """Summarize how tests should be improved and persist as EvalReflection."""
    run_ctx = state.run or RunContext()

    latest_score = run_ctx.score_history[-1] if run_ctx.score_history else run_ctx.score
    attempt_number = run_ctx.attempts + 1

    failed_cases = list(run_ctx.last_failed_cases or [])
    recent_failures = failed_cases[:5]

    if recent_failures:
        failure_lines: List[str] = []
        for case in recent_failures:
            failure_lines.append(
                (
                    f"  Input: {_truncate(case.get('input'))}\n"
                    f"  Expected: {_truncate(case.get('expected_output'))}\n"
                    f"  Actual: {_truncate(case.get('actual_output')) or '(empty)'}\n"
                    f"  Judge feedback: {_truncate(case.get('judge_comment')) or '(none)'}\n"
                    f"  Score: {case.get('score', 0.0):.3f}"
                )
            )
        failures_text = "\n".join(failure_lines)
    else:
        failures_text = "All tests passed on the latest attempt. Focus on preventing regressions."

    score_history = ", ".join(f"{score:.3f}" for score in (run_ctx.score_history or [])) or "(no score history)"

    summary_prompt = (
        "You are a QA lead improving end-to-end eval tests.\n"
        "Review the provided run context and produce an EvalReflection with actionable guidance.\n"
        "Keep recommendations concise, specific, and test-focused.\n\n"
        f"User expectations: {state.user_context.user_expectation}\n"
        f"Attempt number: {attempt_number}\n"
        f"Latest aggregate score: {latest_score:.3f}\n"
        f"Score history: {score_history}\n"
        "Failed test cases (up to 5 most recent):\n"
        f"{failures_text}\n\n"
        "Return fields that match the EvalReflection schema.\n"
        "- summary: 2-3 sentences describing key gaps.\n"
        "- failure_modes: list of short phrases (max 4) capturing root causes.\n"
        "- recommended_tests: list of concrete future test ideas (max 5).\n"
        "- latest_score, attempt, dataset_name, experiment_name should reflect the context above."
    )

    reflection_llm = LLM.with_structured_output(EvalReflection)
    reflection: EvalReflection = reflection_llm.invoke(summary_prompt)

    # Ensure correct agent_name populated
    reflection.agent_name = state.github_context.agent_name
    reflection.latest_score = reflection.latest_score or latest_score
    reflection.attempt = reflection.attempt or attempt_number

    logger.info(
        "reflect_node: captured reflection trace (agent=%s prompt_chars=%d failures_used=%d)",
        state.github_context.agent_name,
        len(summary_prompt),
        len(recent_failures),
    )

    persist_reflection(state.github_context.agent_name, reflection)

    updated_run = run_ctx.model_copy(
        update={
            "attempts": attempt_number,
        }
    )

    return {
        "run": updated_run,
        "codex_thread_id": state.codex_thread_id,
        "codex_request": state.codex_request,
        "codex_response": state.codex_response,
    }


