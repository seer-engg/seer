from shared.logger import get_logger

from agents.codex.state import CodexState
from langchain_core.messages import HumanMessage, BaseMessage
from shared.llm import get_llm, get_agent_final_respone
from agents.codex.format_thread import fetch_thread_timeline_as_string
from langchain_core.messages import SystemMessage
from shared.config import config
from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import (
    REMOVE_ALL_MESSAGES,
)
logger = get_logger("codex.nodes.reflect")

SYSTEM_PROMPT = """
You are an experienced AI engineer , with years of experience in building langchain/langgraph based AI agents.

## Role
your role is to analyse given eval cases and thread traces and provide your  analysis of why agent is failing evals. Only provide error analysis not any fixes.

"""

EVALS_AND_THREAD_TRACE_TEMPLATE = """    
<EVAL> 
{eval}
</EVAL>

<THREAD TRACE>
{thread_trace}
</THREAD TRACE>
"""

USER_PROMPT = """
Below are the latest failed evals and their thread traces:
<LATEST FAILED EVALS AND THREAD TRACES>
{latest_failed_evals_and_thread_traces}
</LATEST FAILED EVALS AND THREAD TRACES>

Please provide a  analysis of why agent is failing evals along with any errors that are visible in the thread traces.

## Output
- errors_verbatim: The errors as they are written in the thread trace.
- agent_analysis: Why agent is failing evals ? write  bullet points heighliting what is wrong with the agent.

"""

from typing import List
from pydantic import BaseModel, Field
class FailureAnalysis(BaseModel):
    errors_verbatim: List[str] = Field(description="The errors as they are written in the thread trace.")
    agent_analysis: List[str] = Field(description="Why agent is failing evals ? write  bullet points heighliting what is wrong with the agent.")

    

async def reflector(state: CodexState) -> CodexState:
    """Reflect on the latest test results and plan necessary fixes."""
    logger.info("Reflecting on failed implementation...")
    
    experiment_results = state.latest_results or state.experiment_context.results

    llm = get_llm().with_structured_output(FailureAnalysis)
    evals_and_thread_traces=[] 
    for eval in experiment_results:
        if eval.passed:
            continue
        x={
            "INPUT:": eval.dataset_example.input_message,
            "EXPECTED OUTPUT:": eval.dataset_example.expected_output.expected_action,
            "ACTUAL OUTPUT:": eval.actual_output,
            "SCORE:": eval.score,
            "JUDGE FEEDBACK:": eval.judge_reasoning
        }
        thread_trace = await fetch_thread_timeline_as_string(eval.thread_id, config.langfuse_project_name)
        evals_and_thread_traces.append(
            EVALS_AND_THREAD_TRACE_TEMPLATE.format(
                eval=x,
                thread_trace=thread_trace
            )
        )

    input_messages = []
    input_messages.append(SystemMessage(content=SYSTEM_PROMPT))
    input_messages.append(HumanMessage(content=USER_PROMPT.format(latest_failed_evals_and_thread_traces=evals_and_thread_traces)))
    response: FailureAnalysis = await llm.ainvoke(input_messages)

    reflection = f"""
    My agent is failing evals because of the following reasons:

    ## Failure Reasons
    
    {response.agent_analysis}
    """
    if response.errors_verbatim:
        reflection += f"""
        ## Errors Verbatim from the thread trace
        {response.errors_verbatim}
        """
    reflection += """
    Please fix my agent 

    """
        
    logger.info(f"Reflection complete. New instruction: {reflection[:100]}...")
    
    # We return a HumanMessage, which will be the *input* for the implement_task_plan agent
    return {
        "developer_thread": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES), 
            HumanMessage(content=reflection)],
        "attempt_number": state.attempt_number + 1,
    }
