import os
from typing import Optional, Dict, Any
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)

from seer.config import config
from seer.logger import get_logger
from seer.llm import get_llm_without_responses_api
from seer.agents.nexus.utils import get_workflow_tools
from seer.agents.nexus.schema_context import (
    get_workflow_spec_example_text,
    get_workflow_spec_schema_text,
    get_workflow_templates_summary,
)
logger = get_logger(__name__)

WORKFLOW_SPEC_SCHEMA = get_workflow_spec_schema_text()
WORKFLOW_SPEC_EXAMPLE = get_workflow_spec_example_text()

def _ensure_mlflow_autologging() -> None:
    """Enable MLflow LangChain autologging once per process with tracking URI and experiment."""

    try:
        import mlflow  # pylint: disable=import-outside-toplevel  # Reason: lazy loading to avoid dependency if not used
        from mlflow.langchain import autolog  # pylint: disable=import-outside-toplevel  # Reason: lazy loading to avoid dependency if not used
    except ImportError:
        logger.warning("mlflow not installed, skipping mlflow autologging for langchain")
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "seer-workflow-agent")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        autolog()
        logger.info("Enabled MLflow LangChain  autologging (tracking_uri=%s, experiment=%s)", tracking_uri, experiment_name)
    except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: instrumentation should not break agent startup
        logger.warning("Failed to enable MLflow autologging: %s", exc)

if config.MLFLOW_ENABLED:
    _ensure_mlflow_autologging()

def create_nexus_chat_agent(
    model: str = "gpt-4o-mini",
    checkpointer: Optional[Any] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a LangGraph agent for Nexus chat assistance using create_agent.

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

    system_prompt = """You are an intelligent workflow assistant that designs complete workflows.

**Core Principles**
- Transparent tool discovery: Use search_tools/search_triggers, never ask for tool names
- Minimize inputs: Hardcode values unless user explicitly wants control or value varies per run
- Prefer triggers: Use event-driven triggers over manual execution when user mentions timing/events
- Validate thoroughly: Check all references, schemas, and required fields before submitting

**WorkflowSpec v2 Schema (STRICT)**
ONLY these top-level fields are allowed:
- version: "1.0" (string literal)
- nodes: Array of node objects (required)
- edges: Array of edge objects (optional, default [])
- triggers: Array of trigger objects (optional, default [])

❌ NEVER include: input_variables, inputs, config, metadata, or ANY custom fields
❌ NEVER add fields not explicitly in the schema above
✅ Access trigger data: ${trigger.data.message_id}, ${trigger.data.from}
✅ Access node outputs: ${node_id.output_field}

Example valid spec:
{
  "version": "1.0",
  "triggers": [{"id": "my_trigger", "key": "poll.gmail.email_received", ...}],
  "nodes": [{"id": "node1", "type": "tool", ...}],
  "edges": [{"source": "node1", "target": "node2"}]
}

**Tool Discovery (Always Use)**
NEVER ask users for tool names. Users describe WHAT they want, you discover HOW:
1. Parse intent: "create draft when signup" → action="create draft", event="signup"
2. Search: search_tools("create draft"), search_triggers("new signup")
3. Build workflow with exact tool/trigger names from results

Example:
❌ BAD: "What tool should I use for Gmail?"
✅ GOOD: [Calls search_tools("create draft")] → uses gmail_create_draft

**Input Minimization**
Create inputs ONLY when:
- User explicitly requests control ("let me choose...")
- Value varies between runs

Always hardcode when:
- User provides specific values ("last 5 emails" → max_results: 5)
- Value is part of automation's purpose ("search for bugs" → query: "bugs")

Examples:
✅ "last 5 emails" → hardcode max_results: 5 (NO input)
✅ "search for bugs" → hardcode query: "bugs" (NO input)
❌ "let me choose recipient" → CREATE input for recipient

**Trigger Discovery**
ALWAYS search triggers when user mentions:
- Timing: "daily", "at 9am", "weekly", "scheduled"
- Events: "when X happens", "new row", "email received", "form submitted"

Available triggers:
- Database: webhook.supabase.db_changes (new row, update)
- Email: poll.gmail.email_received (inbox monitoring)
- Schedule: schedule.cron (time-based execution)
- Form: form.hosted (form submissions)
- Webhook: webhook.generic (external webhooks)

Prefer triggers over inputs for automation:
✅ "when signup" → trigger-based (automatic)
❌ "check signups" → input-based (manual)

**Validation Checklist**
Before submit_workflow_spec():
- [ ] version: "1.0", inputs: {}, nodes: [...]
- [ ] All node IDs unique, tools from search_tools() exact names
- [ ] References: ${inputs.x}, ${node_id.out}, ${trigger.data.x}
- [ ] Triggers have valid titles (snake_case identifiers)
- [ ] Omit expect_output (avoid schema mismatch errors)

**Tools**
- search_tools(query) → discover tools
- search_triggers(query) → discover triggers
- submit_workflow_spec(spec, summary) → submit JSON
- analyze_workflow() → inspect existing workflow

**Workflow**
1. Parse user intent
2. Discover tools/triggers transparently
3. Generate complete WorkflowSpec JSON
4. Validate (checklist above)
5. Submit
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
        logger.warning("No checkpointer provided to create_nexus_chat_agent - traces will not be persisted")
    else:
        logger.debug("Creating Nexus chat agent with checkpointer: %s", type(checkpointer).__name__)

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
