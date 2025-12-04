from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.schema import ExperimentResultContext, FailureAnalysis
from shared.llm import get_llm, get_agent_final_respone
from langchain_core.messages import HumanMessage
from shared.config import config


logger = get_logger("eval_agent.execute.prepare_result")

USER_PROMPT = """
Based on this output text prepare the FailureAnalysis object.

Output text:
{output_text}

"""

async def prepare_result_node(state: TestExecutionState) -> dict:

    llm = get_llm()

    prompt = USER_PROMPT.format(output_text=state.assertion_output)

    structured_llm = llm.with_structured_output(FailureAnalysis, method="json_schema", strict=True)

    failure_analysis: FailureAnalysis = await structured_llm.ainvoke(input=[HumanMessage(content=prompt)])


    started_at = state.started_at 
    completed_at = state.completed_at
    example = state.dataset_example


    # Build final result object
    result = ExperimentResultContext(
        thread_id=(state.thread_id or f"unknown_{example.example_id}"),
        dataset_example=example,
        actual_output=(state.agent_output or ""),
        analysis=failure_analysis,
        passed=(failure_analysis.score >= config.eval_pass_threshold),
        started_at=started_at,
        completed_at=completed_at,
    )

    return {
        "result": result,
    }