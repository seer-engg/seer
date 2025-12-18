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

   
def compile_graph(workflow: StateGraph):
    # Initialize checkpointer for human-in-the-loop interrupts
    checkpointer = None
    if config.database_uri:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            logger.info(f"Initializing PostgresSaver checkpointer with database URI")
            checkpointer = PostgresSaver.from_conn_string(config.database_uri)
            # Setup tables on first run (idempotent)
            try:
                checkpointer.setup()
                logger.info("PostgresSaver checkpointer setup complete")
            except Exception as e:
                # Tables might already exist, which is fine
                logger.debug(f"PostgresSaver setup (tables may already exist): {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgresSaver checkpointer: {e}. Interrupts will not work.")
            logger.warning("Set DATABASE_URI environment variable to enable human-in-the-loop interrupts.")
    else:
        logger.warning("DATABASE_URI not set. Human-in-the-loop interrupts will not work.")
        logger.warning("Set DATABASE_URI environment variable to enable interrupts.")

    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    # Configure Langfuse callbacks at graph compilation time for LangGraph dev server
    # This ensures traces are created even when graph is invoked via HTTP
    # CRITICAL: For LangGraph dev server, callbacks MUST be configured at compile time
    # using .with_config(), not just passed when invoking via RemoteGraph
    # See: https://langfuse.com/guides/cookbook/integration_langgraph
    from agents.eval_agent.constants import LANGFUSE_CLIENT
    if LANGFUSE_CLIENT and config.langfuse_public_key:
        try:
            from langfuse.langchain import CallbackHandler
            from langfuse import propagate_attributes
            
            # Create a custom wrapper that adds metadata to the root trace
            # CRITICAL: For LangGraph dev server, metadata must be set when the root chain starts
            class MetadataCallbackHandler(CallbackHandler):
                def __init__(self, *args, metadata=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.metadata = metadata or {}
                    self._root_chain_started = False
                
                def on_chain_start(self, serialized, inputs, **kwargs):
                    """Override to add metadata when LangGraph root chain starts"""
                    result = super().on_chain_start(serialized, inputs, **kwargs)
                    
                    # Detect LangGraph root chain start (name is "LangGraph" or id is None)
                    run_id = kwargs.get("run_id")
                    parent_run_id = kwargs.get("parent_run_id")
                    
                    # Root chain has no parent_run_id and is the LangGraph chain
                    if not self._root_chain_started and parent_run_id is None:
                        self._root_chain_started = True
                        from langfuse import get_client, propagate_attributes
                        langfuse = get_client()
                        try:
                            # Use propagate_attributes to ensure metadata is attached to root trace
                            # This is the same approach Supervisor uses
                            with propagate_attributes(metadata=self.metadata):
                                # The metadata will now be propagated to all observations
                                # Update the trace directly as well for immediate effect
                                langfuse.update_current_trace(metadata=self.metadata)
                                logger.debug(f"✅ Added metadata to root trace: {self.metadata}")
                        except Exception as e:
                            logger.warning(f"Failed to add metadata to root trace: {e}")
                    
                    return result
            
            langfuse_handler = MetadataCallbackHandler(
                public_key=config.langfuse_public_key,
                metadata={"project_name": config.project_name}
            )
            
            # Use with_config to attach callbacks at compile time
            # This is required for LangGraph dev server to capture traces
            compiled_graph = compiled_graph.with_config({
                "callbacks": [langfuse_handler]
            })
            
            logger.info(f"📊 Langfuse tracing configured at graph compilation time with project_name={config.project_name}")
        except Exception as e:
            logger.warning(f"Failed to configure Langfuse callbacks at graph compilation: {e}")
    
    return compiled_graph


graph = compile_graph(build_graph())
