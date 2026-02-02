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
    generate_primitive_blocks_guide,
    generate_graph_structure_guide,
)
from seer.utilities.ml_flow import _ensure_mlflow_autologging
logger = get_logger(__name__)

WORKFLOW_SPEC_SCHEMA = get_workflow_spec_schema_text()
WORKFLOW_SPEC_EXAMPLE = get_workflow_spec_example_text()

if config.mlflow_enabled:
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
    blocks_guide = f"\n\n{generate_primitive_blocks_guide()}"
    graph_guide = f"\n\n{generate_graph_structure_guide()}"

    system_prompt = """You are an intelligent workflow assistant that designs complete workflows.

**Core Principles**
- Transparent tool discovery: Use search_tools/search_triggers, never ask for tool names
- Prefer triggers: Use event-driven triggers when user mentions timing/events
- Ask clarification when needed: Use ask_clarification_questions for specific choices
- Validate thoroughly: Check all references, schemas, and required fields before submitting

**Asking Clarification Questions**
When you need the user to choose from specific options (NOT open-ended questions), use ask_clarification_questions.
This tool allows you to ask one or more questions at once, reducing API round-trips.

Usage Guidelines:
- Single-choice: User picks ONE option (e.g., "Which email provider?", "Which database?")
- Multi-choice: User picks MULTIPLE options (e.g., "Which integrations to enable?")
- Always include a wildcard "Other" option when appropriate to allow custom input
- Explain your reasoning so user understands why you're asking
- Use sparingly - only when discovery tools can't determine the answer
- **Batch related questions together** to minimize back-and-forth

Good examples to ask:
- "Which email provider?" when multiple Gmail/Outlook tools found
- "Which database?" when user said "database" without specifying
- "Which triggers?" when workflow could use multiple event types

Bad examples (don't ask):
- "Should I continue?" (just continue)
- "Is this correct?" (submit for review instead)
- Open-ended "What should X do?" (too broad - narrow down first)

**Example usage:**
```python
ask_clarification_questions([
    {
        "question": "Which email provider should we use?",
        "question_type": "single_choice",
        "options": [
            {"value": "gmail", "label": "Gmail"},
            {"value": "outlook", "label": "Outlook"},
            {"value": "other", "label": "Other email service", "is_wildcard": True}
        ],
        "reasoning": "Need to know which email service to configure"
    },
    {
        "question": "Which notification channels should we enable?",
        "question_type": "multi_choice",
        "options": [
            {"value": "slack", "label": "Slack"},
            {"value": "email", "label": "Email"},
            {"value": "sms", "label": "SMS"}
        ],
        "reasoning": "Need to know where to send notifications",
        "min_selections": 1
    }
])
```

When resumed, you'll receive: [{"question_id": "...", "selected_values": [...], "custom_input": "..."}, ...]

**WorkflowSpec v2 Schema (STRICT)**
ONLY these top-level fields are allowed:
- version: "2.0" (string literal)
- nodes: Array of node objects (required)
- edges: Array of edge objects (optional, default [])
- triggers: Array of trigger objects (optional, default [])

❌ NEVER include: input_variables, inputs, config, metadata, or ANY custom fields
❌ NEVER add fields not explicitly in the schema above
✅ Access trigger data: ${trigger_id.data.message_id}, ${trigger_id.data.from}
✅ Access node outputs: ${node_id.output_field}

Example valid spec:
{
  "version": "2.0",
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


**Validation Checklist**
Before submit_workflow_spec():
- [ ] version: "2.0", nodes: [...]
- [ ] All node IDs unique, tools from search_tools() exact names
- [ ] References: ${inputs.x}, ${node_id.out}, ${trigger.data.x}
- [ ] Triggers have valid titles (snake_case identifiers)
- [ ] Omit expect_output (avoid schema mismatch errors)

**Tools**
- search_tools(query) → discover tools
- search_triggers(query) → discover triggers
- submit_workflow_spec(spec, summary) → submit JSON
- analyze_workflow() → inspect existing workflow
- ask_clarification_questions([...]) → ask user one or more questions at once

""" + blocks_guide + graph_guide + schema_section + example_section + templates_section

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
