from shared.logger import get_logger

from agents.codex.state import CodexState
from langchain_core.messages import HumanMessage
from shared.llm import get_llm
from agents.codex.format_thread import fetch_thread_timeline_as_string
from sandbox.constants import TARGET_AGENT_LANGSMITH_PROJECT
from langchain_core.messages import SystemMessage

logger = get_logger("codex.nodes.reflect")

SYSTEM_PROMPT = """
You are a Refective person , part of an agent development team.
Your team is trying to enhance an agent so that it can pass all the eval cases. Based on some failed eval cases your team devised a plan to fix the agent. A programmer implemented some code changes to the agent following the plan.
But after the implementation, the agent is not passing all the eval cases. You are tasked to reflect on the latest test results and suggest necessary policy changes the plan should follow to fix the agent.
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

Please provide a new, concrete set of implementation steps to fix the remaining issues.
"""

async def reflector(state: CodexState) -> CodexState:
    """Reflect on the latest test results and plan necessary fixes."""
    logger.info("Reflecting on failed implementation...")
    if not state.latest_test_results:
        logger.warning("No test results available for reflection; skipping prompt generation")
        return {
            "messages": [HumanMessage(content="No test results found, attempting plan again.")]
        }

    llm = get_llm(model="codex")
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
    if isinstance(response.content, list):
        for content in response.content:
            if content.get("type") == "text":
                reflection += content.get("text")
    else:
        reflection = response.content
        
    logger.info(f"Reflection complete. New instruction: {reflection[:100]}...")
    
    # We return a HumanMessage, which will be the *input* for the implement_task_plan agent
    return {
        "messages": [HumanMessage(content=reflection)],
    }
