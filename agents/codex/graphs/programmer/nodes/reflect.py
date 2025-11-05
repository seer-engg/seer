from shared.logger import get_logger

from agents.codex.common.state import ProgrammerState
from langchain_core.messages import HumanMessage
from agents.codex.llm.model import get_chat_model
from agents.codex.graphs.planner.format_thread import fetch_thread_timeline_as_string
from sandbox.constants import TARGET_AGENT_LANGSMITH_PROJECT
from langchain_core.messages import SystemMessage

logger = get_logger("programmer.reflect")

SYSTEM_PROMPT = """
You are a Refective person , part of an agent development team.
Your team is trying to enhance an agent so that it can pass all the eval cases. Based on some failed eval cases your team devised a plan to fix the agent. A programmer implemented some code changes to the agent following the plan.
But after the implementation, the agent is not passing all the eval cases. You are tasked to reflect on the latest test results and suggest necessary changes.
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
Our Team Devised a Plan to fix the agent:
<plan>
{plan}
</plan>

in order to pass these failing evals :
<failing_evals>
{originalfailing_evals}
</failing_evals>

But after the implementation, the agent is not passing all the eval cases. You are tasked to reflect on the latest test results and suggest necessary changes. Below are the lates failed evals and their thread traces:
<LATEST FAILED EVALS AND THREAD TRACES>
{latest_failed_evals_and_thread_traces}
</LATEST FAILED EVALS AND THREAD TRACES>
"""

async def reflect(state: ProgrammerState) -> ProgrammerState:
    """Reflect on the latest test results and plan necessary fixes."""
    if not state.latest_test_results:
        logger.warning("No test results available for reflection; skipping prompt generation")
        return {}

    llm = get_chat_model()
    evals_and_thread_traces=[] 
    for eval in state.latest_test_results:
        if eval.passed:
            continue
        x={
            "INPUT:": eval.dataset_example.input_message,
            "EXPECTED OUTPUT:": eval.dataset_example.expected_output,
            "ACTUAL OUTPUT:": eval.actual_output,
            "SCORE:": eval.score,
            "JUDGE FEEDBACK:": eval.judge_reasoning
        }
        thread_trace = await fetch_thread_timeline_as_string(eval.thread_id, TARGET_AGENT_LANGSMITH_PROJECT)
        evals_and_thread_traces.append(
            EVALS_AND_THREAD_TRACE_TEMPLATE.format(
                eval=x,
                thread_trace=thread_trace
            )
        )
    originalfailing_evals = []
    for eval in state.experiment_context.results:
        if eval.passed:
            continue
        x={
            "INPUT:": eval.dataset_example.input_message,
            "EXPECTED OUTPUT:": eval.dataset_example.expected_output,
            "ACTUAL OUTPUT:": eval.actual_output,
            "SCORE:": eval.score,
            "JUDGE FEEDBACK:": eval.judge_reasoning
        }
        originalfailing_evals.append(x)

    input_messages = []
    input_messages.append(SystemMessage(content=SYSTEM_PROMPT))
    input_messages.append(HumanMessage(content=USER_PROMPT.format(plan=state.taskPlan, originalfailing_evals=originalfailing_evals, latest_failed_evals_and_thread_traces=evals_and_thread_traces)))
    response = await llm.ainvoke(input_messages)

    reflection = ""
    for content in response.content:
        if content.get("type") == "text":
            reflection += content.get("text")
    return {
        "messages": [HumanMessage(content=reflection)],
    }