from typing import Optional, Dict, Any
from shared.logger import get_logger
from shared.llm import get_llm_without_responses_api
from agents.workflow_agent.utils import get_workflow_tools
logger = get_logger(__name__)

from langchain.agents import create_agent
from langchain.agents.middleware import (
        SummarizationMiddleware,
    )
import json
# Autolog LangChain
import mlflow
mlflow.langchain.autolog()
EXAMPLE= {
  "name": "collab 1",
  "graph_data": {
    "edges": [
      {
        "id": "edge-block-cbca008b-block-80263eb2",
        "source": "block-cbca008b",
        "target": "block-80263eb2",
      },
      {
        "id": "edge-block-80263eb2-block-5f78e25a",
        "source": "block-80263eb2",
        "target": "block-5f78e25a",
      },
      {
        "id": "edge-block-5f78e25a-block-68c090f7",
        "source": "block-5f78e25a",
        "target": "block-68c090f7",
      }
    ],
    "nodes": [
      {
        "id": "block-cbca008b",
        "data": {
          "type": "tool",
          "label": "Read Gmail emails (last month, collab)",
          "config": {
            "params": {
              "q": "collab newer_than:30d",
              "label_ids": [
                "INBOX"
              ],
              "max_results": 2,
              "include_body": True
            },
            "tool_name": "gmail_read_emails"
          },
        },
        "type": "tool",
      },
      {
        "id": "block-80263eb2",
        "data": {
          "type": "for_loop",
          "label": "For each email",
          "config": {
            "list": "{{read_gmail_emails_last_month_collab.output}}",
            "item_var": "item",
            "array_mode": "variable",
            "input_refs": {},
            "array_literal": [],
            "array_variable": "{{read_gmail_emails_last_month_collab.output}}"
          },
        },
        "type": "for_loop",
      },
      {
        "id": "block-68c090f7",
        "data": {
          "type": "tool",
          "label": "Create Gmail draft (reply in thread)",
          "config": {
            "params": {
              "to": "[{{summarize_draft_reply.output.to_email}}]",
              "subject": "Re: {{for_each_email.item.subject}}",
              "body_text": "Summary (for you):{{summarize_draft_reply.output.email_body}}",
              "thread_id": "{{for_each_email.item.thread_id}}",
              "in_reply_to": "{{for_each_email.item.message_id}}"
            },
            "tool_name": "gmail_create_draft",
            "input_refs": {}
          },
        },
        "type": "tool",
      },
      {
        "id": "block-5f78e25a",
        "data": {
          "type": "llm",
          "label": "Summarize + draft reply",
          "config": {
            "model": "gpt-5-mini",
            "input_refs": {},
            "temperature": 0,
            "user_prompt": "You are an email assistant.\n\nGiven this email, produce:\n1) A 3-5 bullet summary of what the sender wants.\n2) A concise proposed reply draft (professional, friendly), that:\n   - Answers questions if possible\n   - Suggests next steps for collaboration\n   - Includes 2-3 concrete time slots suggestions (generic, no timezone unless present)\n   - Keeps it under 150 words unless necessary\n\nReturn valid JSON with keys:\nsummary_bullets (array of strings)\ndraft_body_text (string)\n\nEmail:\nFrom: {{for_each_email.item.from}}\n\nTo: {{for_each_email.item.to}}\nSubject: {{for_each_email.item.subject}}\nDate: {{for_each_email.item.date}}\nBody:\n{{for_each_email.item.body_text}}\n",
            "output_schema": {
              "type": "object",
              "properties": {
                "to_email": {
                  "type": "string",
                  "description": "draft receipient email"
                },
                "email_body": {
                  "type": "string",
                  "description": "body of email"
                }
              }
            },
            "system_prompt": "Email Assistant",
            "response_format": "json"
          },
        },
        "type": "llm",
      }
    ]
  },
}

EXAMPLE_WORKFLOW = json.dumps(EXAMPLE, indent=2)

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
    workflow_context = ""
    template_hint_section = ""
    if workflow_state:
        workflow_context = f"\n\nCurrent workflow state:\n{json.dumps(workflow_state, indent=2)}\n\nUse this information when calling tools. Tools automatically access workflow state from thread context via runtime configuration."
        alias_examples = workflow_state.get("template_reference_examples") or {}
        if alias_examples:
            alias_lines = []
            for block_id, examples in alias_examples.items():
                if not examples:
                    continue
                alias_lines.append(f"- {block_id}: {', '.join(examples)}")
            if alias_lines:
                template_hint_section = "\nTemplate reference hints (use these names when writing {{alias.output}} expressions):\n" + "\n".join(alias_lines)
    
    system_prompt = f"""You are an intelligent workflow assistant that helps users build and edit workflows. Your role is to understand user intent, discover appropriate tools, and create workflows that achieve their goals.
**Core Principles:**
- Ask questions in everyday language - avoid jargon and technical terms
- Use search_tools to discover tools dynamically - let the LLM reason about tool selection
- When referencing other blocks, use the alias hints returned by add_workflow_block tools
- Always use the aliases from tool responses - never invent template variable names
- Remember alias names are always between double curly braces like {{alias.output}} 
- Always think through your reasoning and provide clear explanations for suggestions.{template_hint_section}{workflow_context}
- always review that you are using coorect aliases returned by the add_workflow_block tool  tool.

**Creating Workflows:**
1. Understand user intent - what outcome do they want?
2. If unclear, ask 1-2 clarifying questions in plain language (e.g., "What should happen if no emails are found?")
3. Search for relevant tools using search_tools(query, reasoning)
4. Build the workflow step-by-step: add blocks, connect them via tool add_workflow_edge, validate intent
5. remember to connect blocks to each other via tool add_workflow_edge after you have added all blocks

**Function Blocks Quick Reference:**
- LLM block: Call a language model by providing at least a `user_prompt`. Optional `system_prompt` adds steering context, `model` chooses among gpt-5 mini/nano/full or gpt-4o, `temperature` (0-2) controls randomness, and `output_schema` enables structured responses.
- For Loop block: Iterate over each element of an array. `array_mode` picks between variable-driven (`array_variable` reference) or manual `array_literal` values, and `item_var` names the value exposed inside the loop. Loop branch runs per item before continuing down the exit branch.

# important point to remember:
- gmail_read_emails doen't return proper email address for the from_email field. it is better to ask llm to generate the email address from the from_email field.

## an example workflow that works perfectly fine
""" + EXAMPLE_WORKFLOW 

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

