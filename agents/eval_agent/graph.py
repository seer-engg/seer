"""overall graph for the eval agent"""
from typing import Literal, Optional
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.nodes.plan import build_plan_subgraph
from agents.eval_agent.nodes.reflect.graph import build_reflect_subgraph
from agents.eval_agent.nodes.allignment.graph import build_alignment_subgraph
from shared.logger import get_logger
from agents.eval_agent.nodes.testing.graph import build_testing_subgraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from shared.config import config

# Enable MLflow tracing only if configured as the active provider
if config.is_mlflow_tracing_enabled:
    import mlflow
    mlflow.langchain.autolog()


logger = get_logger("eval_agent.graph")

def should_continue(state: EvalAgentState) -> Literal["plan", "finalize"]:
    """Determine if the eval loop should continue plan or finalize."""
    return "plan" if state.attempts < config.eval_n_rounds else "finalize"

SYSTEM_PROMPT = """
You are a supervisor for the evaluation agent.
You are responsible for overseeing the evaluation agent flow and ensuring it is working smoothly.

## Evaluation Agent Flow
- alignment: alignement of agent spec with user request
- plan: generation of test cases
- testing: execution of test cases
- finalize: finalization of the evaluation and hadoff to codex

you should analyse the recent conversation and decide the next step.

"""

class SupervisorDecision(BaseModel):
    """Decision for the supervisor."""
    next_step: Literal["alignment", "plan", "testing", "finalize"] = Field(description="The next step to take in the evaluation agent flow")
    reasoning: str = Field(description="The reasoning for the next step")

async def supervisor(state: EvalAgentState) -> dict:
    """Supervisor node for the evaluation agent."""
    # llm = ChatOpenAI(model="gpt-5-mini")
    # llm = llm.with_structured_output(SupervisorDecision)
    # # Only pass through conversational turns; tool chatter can bias routing decisions.
    # filtered_messages = [
    #     m for m in state.messages
    #     if isinstance(m, (HumanMessage, AIMessage))
    # ]
    # input_messages = [SystemMessage(content=SYSTEM_PROMPT)] + filtered_messages
    # response: SupervisorDecision = await llm.ainvoke(input_messages)
    next_step = state.step
    return next_step

def build_graph() -> StateGraph:
    """Build the evaluation agent graph."""
    workflow = StateGraph(EvalAgentState)
    plan_subgraph = build_plan_subgraph()
    alignment_subgraph = build_alignment_subgraph()
    testing_subgraph = build_testing_subgraph()
    reflect_subgraph = build_reflect_subgraph()

    # Intent classification nodes
    workflow.add_node("alignment", alignment_subgraph)
    workflow.add_node("plan", plan_subgraph)

    workflow.add_node("testing", testing_subgraph)
    workflow.add_node("finalize", reflect_subgraph)

    # Start with intent classification, then route
    workflow.add_conditional_edges(START, supervisor, {
        "alignment": "alignment",
        "plan": "plan",
        "testing": "testing",
        "finalize": "finalize",
    })
    workflow.add_edge("alignment", END)
    workflow.add_edge("plan", END)
    workflow.add_edge("testing", END)
    workflow.add_edge("finalize", END)

    return workflow


graph = build_graph().compile() # graph object for langgraph dev server
