"""Supervisor agent graph with checkpointer for human-in-the-loop interrupts."""
from langgraph.graph import END, START, StateGraph

from shared.logger import get_logger
from shared.config import config
from agents.supervisor.state import SupervisorState, SupervisorInput, SupervisorOutput
from agents.supervisor.nodes import supervisor

logger = get_logger("supervisor.graph")


def build_graph() -> StateGraph:
    """Build the supervisor agent graph."""
    workflow = StateGraph(
        state_schema=SupervisorState,
        input=SupervisorInput,
        output=SupervisorOutput
    )
    
    # Add the main supervisor node
    workflow.add_node("supervisor", supervisor)
    
    # Simple linear flow: START -> supervisor -> END
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", END)
    
    return workflow


def compile_graph(workflow: StateGraph):
    """
    Compile the graph with checkpointer for interrupt support.
    
    The checkpointer is required for human-in-the-loop interrupts
    (like file requests) to work properly.
    """
    checkpointer = None
    
    # Initialize PostgreSQL checkpointer if database_uri is configured
    if config.database_uri:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            logger.info("Initializing PostgresSaver checkpointer for supervisor agent")
            checkpointer = PostgresSaver.from_conn_string(config.database_uri)
            
            # Setup tables on first run (idempotent)
            try:
                checkpointer.setup()
                logger.info("PostgresSaver checkpointer setup complete")
            except Exception as e:
                # Tables might already exist, which is fine
                logger.debug(f"PostgresSaver setup (tables may already exist): {e}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize PostgresSaver checkpointer: {e}")
            logger.warning("File request interrupts will not work without a checkpointer.")
    else:
        logger.warning("DATABASE_URI not set. Human-in-the-loop interrupts will not work.")
        logger.warning("Set DATABASE_URI environment variable to enable file request interrupts.")
    
    # Compile the graph with checkpointer
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    # Configure Langfuse callbacks at graph compilation time
    # This ensures traces are created even when graph is invoked via HTTP
    if config.langfuse_public_key:
        try:
            from langfuse.langchain import CallbackHandler
            
            class MetadataCallbackHandler(CallbackHandler):
                """Custom callback handler that adds metadata to traces."""
                
                def __init__(self, *args, metadata=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.metadata = metadata or {}
                    self._root_chain_started = False
                
                def on_chain_start(self, serialized, inputs, **kwargs):
                    """Override to add metadata when LangGraph root chain starts."""
                    result = super().on_chain_start(serialized, inputs, **kwargs)
                    
                    parent_run_id = kwargs.get("parent_run_id")
                    
                    # Root chain has no parent_run_id
                    if not self._root_chain_started and parent_run_id is None:
                        self._root_chain_started = True
                        try:
                            from langfuse import get_client, propagate_attributes
                            langfuse = get_client()
                            with propagate_attributes(metadata=self.metadata):
                                langfuse.update_current_trace(metadata=self.metadata)
                                logger.debug(f"Added metadata to root trace: {self.metadata}")
                        except Exception as e:
                            logger.warning(f"Failed to add metadata to root trace: {e}")
                    
                    return result
            
            langfuse_handler = MetadataCallbackHandler(
                public_key=config.langfuse_public_key,
                metadata={"agent": "supervisor"}
            )
            
            compiled_graph = compiled_graph.with_config({
                "callbacks": [langfuse_handler]
            })
            
            logger.info("Langfuse tracing configured for supervisor agent")
            
        except Exception as e:
            logger.warning(f"Failed to configure Langfuse callbacks: {e}")
    
    return compiled_graph


# Build and compile the graph
graph = compile_graph(build_graph())

