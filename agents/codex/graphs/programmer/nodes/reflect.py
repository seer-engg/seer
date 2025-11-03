from shared.logger import get_logger

from agents.codex.common.state import ProgrammerState
from langchain_core.messages import HumanMessage

logger = get_logger("programmer.reflect")

PROMPT = """
Recent regression tests failed. Review the failing cases and plan necessary fixes.

<failing_tests>
{test_results}
</failing_tests>
"""


async def reflect(state: ProgrammerState) -> ProgrammerState:
    """Reflect on the latest test results and plan necessary fixes."""
    if not state.latest_test_results:
        logger.warning("No test results available for reflection; skipping prompt generation")
        return {}

    failing_descriptions = []
    for result in state.latest_test_results:
        if result.passed:
            continue
        failing_descriptions.append(
            (
                f"Example ID: {result.dataset_example.example_id}\n"
                f"Input: {result.dataset_example.input_message}\n"
                f"Expected: {result.dataset_example.expected_output}\n"
                f"Actual: {result.actual_output}\n"
                f"Score: {result.score:.3f}\n"
                f"Judge feedback: {result.judge_reasoning}\n"
            )
        )

    if not failing_descriptions:
        return {}

    prompt = PROMPT.format(test_results="\n\n".join(failing_descriptions))
    human_message = HumanMessage(content=prompt)

    messages = list(state.messages or [])
    messages.append(human_message)

    return {
        "messages": messages,
    }