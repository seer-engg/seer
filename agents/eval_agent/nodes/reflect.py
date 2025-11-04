"""node for reflecting on the latest test results and persisting as EvalReflection"""
from typing import Any, List

from agents.eval_agent.constants import LLM
from agents.eval_agent.models import EvalAgentState, EvalReflection
from agents.eval_agent.reflection_store import persist_reflection
from shared.schema import ExperimentResultContext
from shared.logger import get_logger

logger = get_logger("eval_agent.reflect")

def _truncate(text: Any, limit: int = 280) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (truncated {len(text)} chars)"


def reflect_node(state: EvalAgentState) -> dict:
    """Summarize how tests should be improved and persist as EvalReflection."""

    experiment = state.active_experiment
    if not experiment:
        raise RuntimeError("Cannot reflect without an active experiment.")

    latest_score = experiment.mean_score if experiment.results else 0.0

    latest_results: List[ExperimentResultContext] = state.latest_results or []
    failed_cases = [res for res in latest_results if not res.passed]

    if failed_cases:
        failure_lines: List[str] = []
        for case in failed_cases:
            failure_lines.append(
                (
                    f"  Thread / Example ID: {case.dataset_example.example_id}\n"
                    f"  Input: {_truncate(case.dataset_example.input_message)}\n"
                    f"  Expected: {_truncate(case.dataset_example.expected_output)}\n"
                    f"  Actual: {_truncate(case.actual_output) or '(empty)'}\n"
                    f"  Judge feedback: {_truncate(case.judge_reasoning) or '(none)'}\n"
                    f"  Score: {case.score:.3f}\n"
                )
            )
        failures_text = "\n".join(failure_lines)
    else:
        failures_text = "All tests passed on the latest attempt. Focus on preventing regressions."

    user_expectation = state.user_context.user_expectation if state.user_context else ""

    summary_prompt = (
        "You are a QA lead improving end-to-end eval tests.\n"
        "Review the provided experiment context and produce an EvalReflection with actionable guidance.\n"
        "Keep recommendations concise, specific, and test-focused.\n\n"
        f"User expectations: {user_expectation}\n"
        f"Latest aggregate score: {latest_score:.3f}\n"
        f"Score history: {state.active_experiment.mean_score:.3f}\n"
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

    if state.github_context:
        reflection.agent_name = state.github_context.agent_name
    reflection.latest_score = reflection.latest_score or latest_score
    reflection.dataset_name = state.dataset_context.dataset_name if state.dataset_context else None
    reflection.experiment_name = experiment.experiment_name

    logger.info(
        "reflect_node: captured reflection trace (agent=%s prompt_chars=%d failures_used=%d)",
        state.github_context.agent_name,
        len(summary_prompt),
        len(failed_cases),
    )

    if state.github_context:
        # persist failed cases as evidence
        persist_reflection(
            agent_name=state.github_context.agent_name, 
            reflection=reflection,
            evidence_results=failed_cases,
        )

    return {
        "attempts": state.attempts + 1,
    }
