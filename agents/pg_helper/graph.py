"""Supervisor agent graph with checkpointer for human-in-the-loop interrupts."""
from langgraph.graph import END, START, StateGraph

from shared.logger import get_logger
from shared.config import config
from agents.supervisor.state import SupervisorState, SupervisorInput, SupervisorOutput
from agents.supervisor.nodes import supervisor

# Enable MLflow tracing only if configured as the active provider
if config.is_mlflow_tracing_enabled:
    import mlflow
    mlflow.langchain.autolog()

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
    
    return compiled_graph


# Build and compile the graph
graph = compile_graph(build_graph())

