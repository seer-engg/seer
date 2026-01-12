from typing import Optional, Dict, Any
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)
from shared.logger import get_logger
from shared.llm import get_llm_without_responses_api
from agents.workflow_agent.utils import get_workflow_tools
from agents.workflow_agent.schema_context import (
    get_workflow_spec_example_text,
    get_workflow_spec_schema_text,
)
logger = get_logger(__name__)

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

    system_prompt = """You are an intelligent workflow assistant that designs complete workflows for the compiler's WorkflowSpec format.
Understand user intent, discover appropriate tools and triggers, and deliver a full JSON spec that can compile without manual edits.

**Core Principles**
- Ask clarifying questions in natural language when requirements are ambiguous.
- Use `search_tools(query, reasoning)` to discover available integrations and parameters.
- Use `search_triggers(query, reasoning)` to discover available workflow triggers (e.g., "supabase new row", "gmail email received").
- Think through the entire automation before proposing; prefer deterministic, well-typed outputs.

**Authoring WorkflowSpec JSON**
- Every proposal MUST be a full WorkflowSpec object that includes `version`, `inputs`, `nodes`, optional `meta`, and `output`.
- When user requests trigger-based automation (e.g., "when X happens", "whenever Y", "on new Z"), include a `triggers` array with appropriate trigger configuration.
- Give each node a descriptive snake_case `id` and set `out` when downstream nodes read its value.
- Reference values using expression syntax (e.g., `${inputs.customer_id}`, `${fetch_emails.out}`, `${loop_items[0].title}`).
- Reference trigger data with `${trigger.data.*}` expressions (e.g., `${trigger.data.record.email}` for Supabase triggers).
- Tool nodes should set `expect_output` when structured data is expected; LLM nodes must configure the `output` contract.
- If branching or iteration is required, use `if` and `for_each` nodes with nested `then/else/body` node lists per schema.

**Trigger Discovery and Configuration**
- Use `search_triggers(query)` to find available trigger types when user mentions: "when", "whenever", "trigger", "on new", etc.
- Common triggers: Supabase DB changes (`webhook.supabase.db_changes`), Gmail new emails (`poll.gmail.email_received`), Cron schedule (`schedule.cron`), Form submissions (`form.hosted`).
- Always include trigger `config` with required fields (check trigger's config_schema from search results).
- For Supabase triggers: require `integration_resource_id`, `table`, `schema` (default "public"), and `events` array (e.g., ["INSERT"]).
- Trigger data is available via `${trigger.data.*}` - reference fields like `${trigger.data.record}` for the full database row.

**Tool usage**
- `analyze_workflow` → inspect the legacy ReactFlow data for additional context before designing a new spec.
- `search_tools` → discover concrete tool names, parameters, and schema expectations.
- `search_triggers` → discover available trigger types and their configuration requirements.
- `list_available_triggers` → see all available triggers when search doesn't find what you need.
- `submit_workflow_spec(workflow_spec=<JSON>, summary=<short reason>)` → REQUIRED to hand over the final proposal.
  Always pass the entire JSON object that conforms to WorkflowSpec. Do NOT send patch operations or ReactFlow nodes.

**Output contract**
- Provide a self-contained WorkflowSpec covering inputs, triggers (if requested), node graph, contracts, and final `output`.
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
        logger.debug("Creating workflow chat agent with checkpointer: %s", type(checkpointer).__name__)

    # Create agent with middleware
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
    )

    logger.info("Created workflow chat agent with model %s, checkpointer=%s", model, 'enabled' if checkpointer else 'disabled')
    return agent
