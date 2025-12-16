from datetime import datetime
from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.schema import ExperimentResultContext, FailureAnalysis
from shared.llm import get_llm, get_agent_final_respone
from langchain_core.messages import HumanMessage, ToolMessage   
from shared.config import config
import uuid


logger = get_logger("eval_agent.execute.prepare_result")

USER_PROMPT = """
Based on this output text prepare the FailureAnalysis object.

Output text:
{output_text}

"""

async def prepare_result_node(state: TestExecutionState) -> dict:

    started_at = state.started_at 
    completed_at = state.completed_at or datetime.utcnow()
    example = state.dataset_example
    
    # Check if provisioning failed (before target agent invocation)
    provisioning_verification = state.provisioning_verification
    if provisioning_verification and not provisioning_verification.get("provisioning_succeeded", True):
        # Provisioning failed - this is an eval agent failure, not target agent failure
        logger.warning(f"Provisioning failed for example {example.example_id if example else 'unknown'}")
        
        failure_analysis = FailureAnalysis(
            score=0.0,
            judge_reasoning=f"Provisioning verification failed. {provisioning_verification.get('verification_reasoning', 'Unknown reason')}. Missing requirements: {provisioning_verification.get('missing_requirements', [])}"
        )
        
        result = ExperimentResultContext(
            thread_id=(state.thread_id or f"unknown_{example.example_id if example else 'unknown'}"),
            dataset_example=example,
            actual_output="",  # No target agent output since we skipped invocation
            analysis=failure_analysis,
            passed=False,  # Provisioning failure = test failed
            started_at=started_at,
            completed_at=completed_at,
        )
        
        return {
            "result": result,
        }
    
    # Normal path: provisioning succeeded, proceed with assertion analysis
    llm = get_llm()

    prompt = USER_PROMPT.format(output_text=state.assertion_output or "")

    structured_llm = llm.with_structured_output(FailureAnalysis, method="json_schema", strict=True)

    failure_analysis: FailureAnalysis = await structured_llm.ainvoke(input=[HumanMessage(content=prompt)])


    # Build final result object
    result = ExperimentResultContext(
        thread_id=(state.thread_id or f"unknown_{example.example_id if example else 'unknown'}"),
        dataset_example=example,
        actual_output=(state.agent_output or ""),
        analysis=failure_analysis,
        passed=(failure_analysis.score >= config.eval_pass_threshold),
        started_at=started_at,
        completed_at=completed_at,
    )

    output_messages = [ToolMessage(content=failure_analysis.model_dump_json(), tool_call_id=str(uuid.uuid4()))]

    return {
        "result": result,
        "messages": output_messages,
    }