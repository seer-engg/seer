from typing import Optional, Dict, Any
from shared.logger import get_logger
from shared.llm import get_llm_without_responses_api
from agents.workflow_agent.utils import get_workflow_tools
from agents.workflow_agent.schema_context import (
    get_workflow_spec_example_text,
    get_workflow_spec_schema_text,
)
logger = get_logger(__name__)

from langchain.agents import create_agent
from langchain.agents.middleware import (
        SummarizationMiddleware,
    )
import json

WORKFLOW_SPEC_SCHEMA = get_workflow_spec_schema_text()
WORKFLOW_SPEC_EXAMPLE = get_workflow_spec_example_text()

def create_workflow_chat_agent(
    model: str = "gpt-4o-mini",
    checkpointer: Optional[Any] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a LangGraph agent for workflow chat assistance using create_agent.
    
    Uses LangChain v1.0+ create_agent with middleware for summarization
    and human-in-the-loop capabilities.
    
    Args:
        model: Model name to use (e.g., 'gpt-5.2', 'gpt-5-mini')
        checkpointer: Optional LangGraph checkpointer for persistence
        
    Returns:
        LangGraph agent compiled with tools and middleware
    """
    
    llm = get_llm_without_responses_api(model=model, temperature=0, api_key=None)
    
    # System prompt for the workflow assistant
    schema_section = f"\n\nWorkflowSpec schema excerpt (trimmed):\n{WORKFLOW_SPEC_SCHEMA}"
    example_section = f"\n\nValid WorkflowSpec example:\n{WORKFLOW_SPEC_EXAMPLE}"

    system_prompt = """You are an intelligent workflow assistant that designs complete workflows for the compiler's WorkflowSpec format. Understand user intent, discover appropriate tools, and deliver a full JSON spec that can compile without manual edits.

**Core Principles**
- Ask clarifying questions in natural language when requirements are ambiguous.
- Use `search_tools(query, reasoning)` to discover available integrations and parameters.
- Think through the entire automation before proposing; prefer deterministic, well-typed outputs.

**Authoring WorkflowSpec JSON**
- Every proposal MUST be a full WorkflowSpec object that includes `version`, `inputs`, `nodes`, optional `meta`, and `output`.
- Give each node a descriptive snake_case `id` and set `out` when downstream nodes read its value.
- Reference values using expression syntax (e.g., `${inputs.customer_id}`, `${fetch_emails.out}`, `${loop_items[0].title}`).
- Tool nodes should set `expect_output` when structured data is expected; LLM nodes must configure the `output` contract.
- If branching or iteration is required, use `if` and `for_each` nodes with nested `then/else/body` node lists per schema.

**Tool usage**
- `analyze_workflow` → inspect the legacy ReactFlow data for additional context before designing a new spec.
- `search_tools` → discover concrete tool names, parameters, and schema expectations.
- `submit_workflow_spec(workflow_spec=<JSON>, summary=<short reason>)` → REQUIRED to hand over the final proposal. Always pass the entire JSON object that conforms to WorkflowSpec. Do NOT send patch operations or ReactFlow nodes.

**Output contract**
- Provide a self-contained WorkflowSpec covering inputs, node graph, contracts, and final `output`.
- Never emit partial patches or mention legacy tools such as add_workflow_block/add_workflow_edge—the new compiler only accepts full specs.
- Keep reasoning concise but precise so reviewers understand tradeoffs.
""" + schema_section + example_section

    # Get workflow tools (with optional workflow_state injection)
    tools = get_workflow_tools(workflow_state=workflow_state)
    
    # Create summarization model (use same model with lower max tokens)
    summarization_model = get_llm_without_responses_api(
        model=model,
        temperature=0,
        api_key=None,
    )
    
    # Build middleware list
    middleware = [
        SummarizationMiddleware(
            model=summarization_model,
            max_tokens_before_summary=256000,  # 256k token limit
        ),
    ]
    
    # Verify checkpointer is provided (required for persistence)
    if checkpointer is None:
        logger.warning("No checkpointer provided to create_workflow_chat_agent - traces will not be persisted")
    else:
        logger.debug(f"Creating workflow chat agent with checkpointer: {type(checkpointer).__name__}")
    
    # Create agent with middleware
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
    )
    
    logger.info(f"Created workflow chat agent with model {model}, checkpointer={'enabled' if checkpointer else 'disabled'}")
    return agent

