from typing import Optional, Dict, Any
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)
from seer.logger import get_logger
from seer.llm import get_llm_without_responses_api
from seer.agents.workflow_agent.utils import get_workflow_tools
from seer.agents.workflow_agent.schema_context import (
    get_workflow_spec_example_text,
    get_workflow_spec_schema_text,
    get_workflow_templates_summary,
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
    templates_section = f"\n\n{get_workflow_templates_summary()}"

    system_prompt = """You are an intelligent workflow assistant that designs complete workflows for the compiler's WorkflowSpec format.
Your job: translate user intent (natural language) into complete, executable WorkflowSpec JSON.

**CRITICAL: Transparent Tool Discovery**
NEVER ask users for tool names like "gmail_create_draft". Users describe WHAT they want, you discover HOW to do it.

Example conversation:
❌ BAD: "What tool should I use for creating a Gmail draft?"
✅ GOOD: [Calls search_tools("create draft") → finds gmail_create_draft] → builds workflow

**Core Principles**
- Ask clarifying questions about requirements, NOT about tool names
- Use `search_tools(query)` to discover tools from natural language (e.g., "create draft", "send email", "insert row")
- Use `search_triggers(query)` to discover triggers from events (e.g., "new row", "email received", "scheduled")
- Think through the entire automation before proposing

**Tool Discovery Workflow**
1. Parse user intent: "create a draft when someone signs up"
   - Action: "create a draft" → search_tools("create draft")
   - Trigger: "when someone signs up" → search_triggers("new signup")

2. Review search results:
   - top_match shows best tool with confidence score
   - alternatives show other options if ambiguous

3. If multiple high-confidence tools:
   - Ask user to clarify (e.g., "I found Gmail and Slack. Which?")
   - Don't mention tool names, mention capabilities

4. Build workflow with discovered tools

**Authoring WorkflowSpec JSON**
- Every proposal MUST include: `version`, `inputs`, `nodes`, optional `meta`, `output`
- Trigger-based workflows: include `triggers` array with config
- Node IDs: descriptive snake_case (e.g., `create_welcome_draft`)
- Reference expressions: `${inputs.x}`, `${node.out}`, `${trigger.data.record}`
- Tool nodes: use exact tool name from search_tools() result
- LLM nodes: configure `output` contract with schema

**Trigger Configuration**
- Supabase: `webhook.supabase.db_changes` with `{integration_resource_id, table, schema, events}`
- Gmail: `poll.gmail.email_received` with optional filters
- Schedule: `schedule.cron` with `{cron_expression, timezone}`
- Trigger data available via `${trigger.data.*}`

**Tool Usage**
- `search_tools(query)` → discover tools from natural language intent
- `search_triggers(query)` → discover triggers from event descriptions
- `submit_workflow_spec(workflow_spec, summary)` → submit final JSON
- `analyze_workflow()` → inspect existing workflow if modifying

**Examples**

User: "create a draft when someone signs up to my app"
You: [search_tools("create draft")] → gmail_create_draft
     [search_triggers("new signup")] → webhook.supabase.db_changes
     → Build workflow with both

User: "send a message when task is done"
You: [search_tools("send message")] → finds gmail_send_email AND slack_post_message
     → "I found tools for Gmail and Slack. Which would you like to use?"

**Output Contract**
- Complete, self-contained WorkflowSpec JSON
- No partial patches or ReactFlow nodes
- Concise reasoning explaining choices

**Using Templates**
- When user intent matches a template pattern, suggest it: "I found a template for this use case..."
- Templates can be customized - use them as starting points
- Templates show best practices for common integrations
""" + schema_section + example_section + templates_section

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
