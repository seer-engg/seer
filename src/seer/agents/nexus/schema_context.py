"""
Helpers for surfacing the canonical workflow compiler schema to the agent.
"""

from __future__ import annotations
# pylint: disable=too-many-lines # Reason: Comprehensive schema documentation requires length
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from seer.core.schema.models import (
    WorkflowSpec,
    EdgeType,
    ToolNode,
    LLMNode,
    IfNode,
    ForEachNode,
    TriggerSpec,
    Edge,
)
from seer.logger import get_logger

logger = get_logger(__name__)

# Keys that keep the schema digestible while conveying the structure.
_SCHEMA_KEYS = ("title", "type", "properties", "required", "definitions", "default")

_WORKFLOW_SPEC_EXAMPLE: Dict[str, Any] = {
    "version": "2",
    "triggers": [],
    "nodes": [
        {
            "id": "fetch_news",
            "type": "tool",
            "tool": "demo.news_search",
            "inputs": {
                "query": "AI automation trends",
                "timeframe_days": 7,
            },
        },
        {
            "id": "summarize",
            "type": "llm",
            "inputs": {
                "model": "gpt-4o-mini",
                "prompt": "Summarize the top 3 recent articles. Use bullet points with source names.",
                "articles": "${fetch_news}",
            },
            "outputs": {
                "mode": "json",
                "schema": {
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "talking_points": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["talking_points"],
                    }
                },
            },
        },
    ],
    "edges": [
        {"source": "fetch_news", "target": "summarize"},
    ],
}

_WORKFLOW_SPEC_TRIGGER_EXAMPLE: Dict[str, Any] = {
    "version": "2",
    "triggers": [
        {
            "id": "new_signup",
            "key": "webhook.supabase.db_changes",
            "mode": "webhook",
            "provider_config": {
                "integration_resource_id": 123,
                "table": "signups",
                "schema": "public",
                "events": ["INSERT"]
            },
            "event_schema": {
                "type": "object",
                "properties": {
                    "record": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "name": {"type": "string"}
                        }
                    }
                },
                "required": ["record"]
            }
        }
    ],
    "nodes": [
        {
            "id": "create_draft",
            "type": "tool",
            "tool": "gmail_create_draft",
            "inputs": {
                "to": ["${new_signup.data.record.email}"],
                "subject": "Welcome!",
                "body_text": "Hi ${new_signup.data.record.name}, welcome to our platform!"
            },
        }
    ],
    "edges": [
        {"source": "new_signup", "target": "create_draft", "type": "trigger"},
    ],
}


@lru_cache(maxsize=1)
def get_workflow_spec_schema() -> Dict[str, Any]:
    """
    Return the compiler JSON schema for WorkflowSpec with only the most relevant keys.
    Cached because pydantic schema generation is relatively expensive.
    """

    schema = WorkflowSpec.model_json_schema()
    return {key: schema.get(key) for key in _SCHEMA_KEYS if key in schema}


def get_workflow_spec_schema_text(max_chars: int = 4000) -> str:
    """
    Render the schema as formatted JSON, optionally truncating for prompt safety.
    """

    schema_text = json.dumps(get_workflow_spec_schema(), indent=2)
    if len(schema_text) > max_chars:
        schema_text = schema_text[: max_chars - 3] + "..."
    return schema_text


def get_workflow_spec_example_text() -> str:
    """
    Provide compact, valid WorkflowSpec examples for the agent to imitate.
    Includes both input-based and trigger-based workflow examples.
    """

    examples_text = "Example 1 (Input-based workflow):\n"
    examples_text += json.dumps(_WORKFLOW_SPEC_EXAMPLE, indent=2)
    examples_text += "\n\nExample 2 (Trigger-based workflow):\n"
    examples_text += json.dumps(_WORKFLOW_SPEC_TRIGGER_EXAMPLE, indent=2)

    return examples_text


@lru_cache(maxsize=1)
def get_workflow_templates() -> List[Dict[str, Any]]:
    """
    Load all workflow templates from the templates directory.
    Templates provide common workflow patterns that the agent can suggest or use as starting points.

    Returns:
        List of template dictionaries with name, description, tags, customization_guide, and spec
    """
    templates_dir = Path(__file__).parent / "templates"
    templates = []

    if not templates_dir.exists():
        logger.warning("Templates directory not found at %s", templates_dir)
        return templates

    for template_file in templates_dir.glob("*.json"):
        try:
            with open(template_file, "r", encoding="utf-8") as file:
                template_data = json.load(file)
                templates.append(template_data)
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Failed to load template %s: %s", template_file.name, exc)
            continue

    logger.info("Loaded %d workflow templates", len(templates))
    return templates


def get_workflow_templates_summary() -> str:
    """
    Generate a concise summary of available workflow templates for the agent system prompt.

    Returns:
        Formatted string listing templates with their descriptions and use cases
    """
    templates = get_workflow_templates()

    if not templates:
        return "No workflow templates available."

    summary_lines = ["## Common Workflow Templates\n"]
    summary_lines.append("You can suggest these templates when they match user intent:\n")

    for idx, template in enumerate(templates, 1):
        name = template.get("name", "Unknown")
        description = template.get("description", "")
        tags = template.get("tags", [])

        summary_lines.append(f"{idx}. **{name}**")
        summary_lines.append(f"   - Description: {description}")
        summary_lines.append(f"   - Use when: {', '.join(tags[:4])}")
        summary_lines.append("")

    return "\n".join(summary_lines)


def _format_node_type_usage_notes(node_type: str) -> List[str]:
    """Generate usage notes for a specific node type."""
    notes_map = {
        "tool": [
            "**Important:** Tool nodes do NOT have an `outputs` field. "
            "Tool output schema is determined by the tool registry."
        ],
        "llm": [
            "**Important:** LLM nodes require `model` and `prompt` in inputs. "
            "Use `outputs` to specify structured JSON output with schema."
        ],
        "if": [
            "**Important:** Condition should be a boolean expression. "
            "Use edges with `type=conditional_true` and `type=conditional_false` for branching."
        ],
        "for_each": [
            "**Important:** Items should evaluate to a list. "
            "Use edges with `type=loop_body` for loop content and `type=loop_exit` to exit."
        ],
    }
    return notes_map.get(node_type, [])


@lru_cache(maxsize=1)
def generate_node_type_reference() -> str:
    """
    Auto-generate node type reference from Pydantic models.

    Extracts documentation from ToolNode, LLMNode, IfNode, ForEachNode
    including required fields, all fields with descriptions, and usage notes.

    Returns:
        Formatted documentation for each node type
    """
    node_types = {
        "tool": (ToolNode, "Execute a tool from the tool registry"),
        "llm": (LLMNode, "AI inference with model configuration"),
        "if": (IfNode, "Conditional branching based on expression"),
        "for_each": (ForEachNode, "Iterate over a list with loop body"),
    }

    lines = []

    for node_type, (model_cls, description) in node_types.items():
        schema = model_cls.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        lines.append(f"### {node_type}")
        lines.append(description)
        lines.append("")
        lines.append("**Required fields:**")

        for field_name in required:
            if field_name in properties:
                prop = properties[field_name]
                field_type = prop.get("type", "any")
                field_desc = prop.get("description", "")
                default = prop.get("default", "")
                default_str = f" (default: {json.dumps(default)})" if default else ""
                lines.append(f"- `{field_name}` ({field_type}){default_str}: {field_desc}")

        lines.append("")
        lines.append("**All fields:**")

        for field_name, prop in properties.items():
            if field_name in ["type", "ui"]:
                continue
            field_type = prop.get("type", prop.get("anyOf", "any"))
            field_desc = prop.get("description", "")
            is_required = field_name in required
            req_str = " (required)" if is_required else " (optional)"

            if isinstance(field_type, list):
                field_type = "/".join(str(t) for t in field_type)
            elif isinstance(field_type, dict):
                field_type = "object"

            lines.append(f"- `{field_name}`{req_str}: {field_desc}")

        usage_notes = _format_node_type_usage_notes(node_type)
        if usage_notes:
            lines.append("")
            lines.extend(usage_notes)

        lines.append("")

    return "\n".join(lines)


@lru_cache(maxsize=1)
def generate_validation_checklist_from_model() -> str:
    """
    Auto-generate validation checklist from WorkflowSpec Pydantic model.

    Extracts requirements from model_json_schema() to create a checklist of
    validation rules for workflow generation.

    Returns:
        Checklist items extracted from schema
    """
    spec_schema = WorkflowSpec.model_json_schema()
    trigger_schema = TriggerSpec.model_json_schema()
    edge_schema = Edge.model_json_schema()

    lines = ["**Workflow Validation Checklist:**", ""]

    spec_props = spec_schema.get("properties", {})

    lines.append("Top-level requirements:")
    lines.append(f"- version must be \"2\" (default: {spec_props.get('version', {}).get('default', 'N/A')})")
    lines.append("- nodes must be a list of node objects")
    lines.append("- edges must be a list of edge objects")
    lines.append("- triggers must be a list of trigger objects")
    lines.append("")

    lines.append("Node requirements:")
    lines.append("- Each node must have unique `id` (string, min 1 char)")
    lines.append("- Each node must have `type` field: tool, llm, if, or for_each")
    lines.append("- Tool nodes: require `tool` (string) and `inputs` (object)")
    lines.append("- LLM nodes: require `inputs` with `model` and `prompt` fields")
    lines.append("- If nodes: require `condition` (boolean expression)")
    lines.append("- ForEach nodes: require `items` (list expression)")
    lines.append("- Tool nodes MUST NOT have `outputs` field (derived from registry)")
    lines.append("- Node outputs accessed via variable syntax: `${node_id}` or `${node_id.field}`")
    lines.append("")

    trigger_props = trigger_schema.get("properties", {})
    trigger_required = trigger_schema.get("required", [])

    lines.append("Trigger requirements:")
    for field in trigger_required:
        if field in trigger_props:
            prop = trigger_props[field]
            field_desc = prop.get("description", "")
            lines.append(f"- {field}: {field_desc}")
    lines.append("")

    edge_props = edge_schema.get("properties", {})
    edge_required = edge_schema.get("required", [])

    lines.append("Edge requirements:")
    for field in edge_required:
        if field in edge_props:
            prop = edge_props[field]
            field_desc = prop.get("description", "")
            lines.append(f"- {field}: {field_desc}")

    edge_types = [e.value for e in EdgeType]
    lines.append(f"- Valid edge types: {', '.join(edge_types)}")
    lines.append("  - default: sequential flow")
    lines.append("  - trigger: from trigger to first node")
    lines.append("  - conditional_true/conditional_false: if node branches")
    lines.append("  - loop_body/loop_exit: for_each loop edges")
    lines.append("")

    return "\n".join(lines)


@lru_cache(maxsize=1)
def generate_trigger_reference() -> str:
    """
    Auto-generate trigger documentation from TriggerSpec model.

    Returns:
        Formatted trigger specification documentation
    """
    schema = TriggerSpec.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    lines = ["### Trigger Specification", ""]
    lines.append("Triggers define when workflows execute. Each trigger must have:")
    lines.append("")

    lines.append("**Required fields:**")
    for field_name in required:
        if field_name in properties:
            prop = properties[field_name]
            field_desc = prop.get("description", "")
            lines.append(f"- `{field_name}`: {field_desc}")

    lines.append("")
    lines.append("**Configuration fields:**")
    for field_name, prop in properties.items():
        if field_name in required or field_name in ["ui_meta"]:
            continue
        field_desc = prop.get("description", "")
        lines.append(f"- `{field_name}`: {field_desc}")

    lines.append("")
    lines.append("**Important:**")
    lines.append("- Trigger `id` must be unique across all triggers in the workflow")
    lines.append("- Use edge with `type=trigger` to connect trigger to first node")
    lines.append("- Trigger data accessed via `${trigger.data}` in downstream nodes")
    lines.append("- Provider-specific config goes in `provider_config` object")
    lines.append("")

    return "\n".join(lines)


@lru_cache(maxsize=1)
def generate_edge_reference() -> str:
    """
    Auto-generate edge type documentation from Edge and EdgeType models.

    Returns:
        Formatted edge specification documentation
    """
    schema = Edge.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    lines = ["### Edge Specification", ""]
    lines.append("Edges connect nodes to define workflow execution flow.")
    lines.append("")

    lines.append("**Required fields:**")
    for field_name in required:
        if field_name in properties:
            prop = properties[field_name]
            field_desc = prop.get("description", "")
            lines.append(f"- `{field_name}`: {field_desc}")

    lines.append("")
    lines.append("**Edge types:**")
    for edge_type in EdgeType:
        lines.append(f"- `{edge_type.value}`: {_get_edge_type_description(edge_type)}")

    lines.append("")
    lines.append("**Important:**")
    lines.append("- Every node must be reachable via edges (no orphaned nodes)")
    lines.append("- Trigger nodes connect to first processing node via `type=trigger` edge")
    lines.append("- If nodes must have both conditional_true and conditional_false edges")
    lines.append("- ForEach nodes must have loop_body and loop_exit edges")
    lines.append("")

    return "\n".join(lines)


def _get_edge_type_description(edge_type: EdgeType) -> str:
    """Helper to provide human-readable descriptions for edge types."""
    descriptions = {
        EdgeType.default: "Sequential flow from source to target",
        EdgeType.trigger: "Entry point from trigger to first node",
        EdgeType.conditional_true: "If condition evaluates to true",
        EdgeType.conditional_false: "If condition evaluates to false",
        EdgeType.loop_body: "Entry into for_each loop body",
        EdgeType.loop_exit: "Exit from for_each loop",
    }
    return descriptions.get(edge_type, "Unknown edge type")


@lru_cache(maxsize=1)
def generate_primitive_blocks_guide() -> str:
    """
    Generate comprehensive guidance for all primitive workflow blocks.

    Provides detailed documentation for each block type including:
    - Purpose and use cases
    - Complete schema with all fields
    - Expression syntax and examples
    - Common patterns and edge configurations

    Returns:
        Formatted markdown guide for primitive blocks
    """
    guide = """# Primitive Workflow Blocks Reference

## Overview
Seer workflows are built from 5 primitive block types: **tool**, **llm**, **mcp**, **if**, and **for_each**.
Each block is a node in the workflow graph, connected by edges that define execution flow.

---

## 1. TOOL BLOCK (`type: "tool"`)

**Purpose:** Execute tools from the tool registry (e.g., Gmail, Slack, database operations)

**Schema:**
```json
{
  "id": "unique_node_id",
  "type": "tool",
  "tool": "tool_name_from_registry",
  "inputs": {
    "param1": "${reference_or_value}",
    "param2": "literal value"
  }
}
```

**Required Fields:**
- `id` (string): Unique identifier for this node
- `type`: Must be `"tool"`
- `tool` (string): Exact tool name from registry (use `search_tools()` to discover)
- `inputs` (object): Parameters for the tool, can include `${...}` expressions

**Important Notes:**
- ❌ Tool nodes MUST NOT have an `outputs` field - output schema comes from tool registry
- ✅ Use `search_tools(query)` to discover available tools and their exact names
- ✅ Tool output is accessed via `${node_id}` or `${node_id.field_name}`

**Example:**
```json
{
  "id": "send_email",
  "type": "tool",
  "tool": "gmail_send_email",
  "inputs": {
    "to": ["${trigger.data.recipient}"],
    "subject": "Re: ${previous_email.subject}",
    "body": "${draft_message}"
  }
}
```

**Common Use Cases:**
- Send emails/messages (Gmail, Slack)
- Database operations (create, read, update)
- External API calls
- File operations

---

## 2. LLM BLOCK (`type: "llm"`)

**Purpose:** AI inference for text generation, classification, extraction, summarization

**Schema:**
```json
{
  "id": "unique_node_id",
  "type": "llm",
  "inputs": {
    "model": "gpt-5-mini",
    "prompt": "Your prompt with ${data} references",
    "temperature": 0.3,
    "max_tokens": 1000
  },
  "outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "field": {"type": "string"}
        },
        "required": ["field"]
      }
    }
  }
}
```

**Required Fields:**
- `id` (string): Unique identifier
- `type`: Must be `"llm"`
- `inputs` (object): Must contain `model` and `prompt`
  - `model` (string): Model ID (e.g., `"gpt-5-mini"`, `"gpt-4o-mini"`)
  - `prompt` (string): Template with `${...}` expressions for dynamic content
  - `temperature` (number, optional): 0.0-1.0, controls randomness
  - `max_tokens` (number, optional): Maximum tokens to generate
- `outputs` (object): Defines output format
  - `mode`: Either `"text"` or `"json"`
  - `schema`: Required if `mode="json"`, contains JSON Schema

**Output Modes:**

1. **Text Mode** (freeform text):
```json
{
  "outputs": {
    "mode": "text"
  }
}
```

2. **JSON Mode** (structured data):
```json
{
  "outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "category": {"type": "string", "enum": ["urgent", "normal"]},
          "summary": {"type": "string"}
        },
        "required": ["category", "summary"]
      }
    }
  }
}
```

**Example:**
```json
{
  "id": "classify_email",
  "type": "llm",
  "inputs": {
    "model": "gpt-5-mini",
    "temperature": 0.3,
    "prompt": "Classify this email:\\n\\nFrom: ${email.from}\\nSubject: ${email.subject}\\nBody: ${email.body}",
    "email_data": "${fetch_email}"
  },
  "outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "urgency": {"type": "string", "enum": ["urgent", "normal", "low"]},
          "category": {"type": "string"},
          "needs_human_review": {"type": "boolean"}
        },
        "required": ["urgency", "category", "needs_human_review"]
      }
    }
  }
}
```

**Common Use Cases:**
- Email/message classification
- Data extraction from unstructured text
- Content generation (drafts, summaries)
- Sentiment analysis
- Decision-making with structured output

---

## 3. MCP BLOCK (`type: "mcp"`)

**Purpose:** Execute tools from Model Context Protocol (MCP) servers (external tool providers)

**Schema:**
```json
{
  "id": "unique_node_id",
  "type": "mcp",
  "server": "http://localhost:8080/mcp",
  "server_type": "http",
  "tool": "tool_name_on_mcp_server",
  "inputs": {
    "param": "${value}"
  },
  "auth": {
    "headers": {
      "Authorization": "Bearer ${secrets.api_key}"
    }
  }
}
```

**Required Fields:**
- `id` (string): Unique identifier
- `type`: Must be `"mcp"`
- `server` (string): MCP server URL or stdio command
- `tool` (string): Tool name on the MCP server

**Optional Fields:**
- `server_type` (string): `"http"` or `"stdio"` (default: `"http"`)
- `inputs` (object): Parameters for the tool
- `auth` (object): Authentication configuration
  - `headers` (object): HTTP headers (for http servers)
  - `env` (object): Environment variables (for stdio servers)
- `expect_outputs` (object): Output validation schema (same as LLM outputs)

**Authentication:**
- Use `${secrets.key_name}` for credential references
- HTTP servers: provide headers
- Stdio servers: provide env variables

**Example:**
```json
{
  "id": "search_external",
  "type": "mcp",
  "server": "http://search-api.example.com/mcp",
  "server_type": "http",
  "tool": "semantic_search",
  "auth": {
    "headers": {
      "X-API-Key": "${secrets.search_api_key}"
    }
  },
  "inputs": {
    "query": "${user_query}",
    "limit": 10
  }
}
```

**Common Use Cases:**
- External API integrations
- Custom tool execution
- Third-party service calls

---

## 4. IF BLOCK (`type: "if"`)

**Purpose:** Conditional branching - execute different paths based on boolean conditions

**Schema:**
```json
{
  "id": "unique_node_id",
  "type": "if",
  "condition": "${expression == 'value'}"
}
```

**Required Fields:**
- `id` (string): Unique identifier
- `type`: Must be `"if"`
- `condition` (string): Boolean expression using `${...}` syntax

**Edge Configuration:**
If nodes **require** two edges:
```json
{
  "source": "if_node_id",
  "target": "node_when_true",
  "type": "conditional_true"
},
{
  "source": "if_node_id",
  "target": "node_when_false",
  "type": "conditional_false"
}
```

**Condition Expression Syntax:**
- Comparison: `${value} == 'expected'`, `${count} > 10`, `${status} != 'pending'`
- Boolean logic: `${flag}`, `!${flag}`, `${a} && ${b}`, `${a} || ${b}`
- Nested fields: `${node.output.nested.field} == 'value'`

**Examples:**

1. **Simple equality check:**
```json
{
  "id": "check_urgency",
  "type": "if",
  "condition": "${classification.urgency == 'urgent'}"
}
```

2. **Numeric comparison:**
```json
{
  "id": "check_count",
  "type": "if",
  "condition": "${results.count} > 0"
}
```

3. **Complex condition:**
```json
{
  "id": "check_criteria",
  "type": "if",
  "condition": "${status == 'active'} && ${priority} > 5"
}
```

**Common Use Cases:**
- Route based on classification results
- Handle different data states
- Error handling paths
- Feature flags

---

## 5. FOR_EACH BLOCK (`type: "for_each"`)

**Purpose:** Iterate over a list, executing the same operations for each item

**Schema:**
```json
{
  "id": "unique_node_id",
  "type": "for_each",
  "items": "${list_expression}",
  "item_var": "current_item",
  "index_var": "current_index"
}
```

**Required Fields:**
- `id` (string): Unique identifier
- `type`: Must be `"for_each"`
- `items` (string): Expression that resolves to a list (e.g., `"${results}"`, `"${trigger.data.items}"`)

**Optional Fields:**
- `item_var` (string): Variable name for current item (default: `"item"`)
- `index_var` (string): Variable name for current index (default: `"index"`)
- `outputs` (object): Schema for aggregated results

**Edge Configuration:**
ForEach nodes **require** specific edge patterns:
```json
{
  "source": "loop_id",
  "target": "process_item",
  "type": "loop_body"
},
{
  "source": "process_item",
  "target": "loop_id",
  "type": "default"
},
{
  "source": "loop_id",
  "target": "next_node",
  "type": "loop_exit"
}
```

**Accessing Loop Variables:**
- Current item: `${item_var}` (or custom name)
- Current index: `${index_var}` (or custom name)
- Example: `${current_item.id}`, `${current_index}`

**Example:**
```json
{
  "nodes": [
    {
      "id": "loop_emails",
      "type": "for_each",
      "items": "${email_list}",
      "item_var": "email",
      "index_var": "idx"
    },
    {
      "id": "send_email",
      "type": "tool",
      "tool": "gmail_send_email",
      "inputs": {
        "to": ["${email.address}"],
        "subject": "Notification #${idx}",
        "body": "Hello ${email.name}"
      }
    }
  ],
  "edges": [
    {"source": "loop_emails", "target": "send_email", "type": "loop_body"},
    {"source": "send_email", "target": "loop_emails", "type": "default"},
    {"source": "loop_emails", "target": "done", "type": "loop_exit"}
  ]
}
```

**Common Use Cases:**
- Process multiple items from a query/search
- Send bulk emails/notifications
- Batch operations
- Data transformation on lists

---

## Expression Syntax Reference

All blocks support `${...}` expressions for dynamic data:

**Trigger Data:**
- `${trigger_id.data.field}` - Access trigger payload
- `${trigger_id.data.nested.value}` - Nested field access

**Node Outputs:**
- `${node_id}` - Full output of a node
- `${node_id.field}` - Specific field from node output
- `${node_id.nested.field}` - Nested field access

**Loop Variables:**
- `${item}` - Current item in for_each loop (or custom `item_var`)
- `${index}` - Current index in for_each loop (or custom `index_var`)

**Secrets (MCP only):**
- `${secrets.api_key}` - Reference to stored credentials

**Operators:**
- Comparison: `==`, `!=`, `>`, `<`, `>=`, `<=`
- Boolean: `&&` (and), `||` (or), `!` (not)
- Array access: `${array[0]}`, `${array[0].field}`

---

## Block Composition Patterns

**Pattern 1: Tool → LLM (Process then Analyze)**
```json
{
  "nodes": [
    {"id": "fetch", "type": "tool", "tool": "gmail_get_message", "inputs": {"message_id": "${trigger.data.id}"}},
    {"id": "analyze", "type": "llm", "inputs": {"model": "gpt-5-mini", "prompt": "Analyze: ${fetch.body}"}, "outputs": {"mode": "json", ...}}
  ],
  "edges": [
    {"source": "fetch", "target": "analyze"}
  ]
}
```

**Pattern 2: LLM → If (Classify then Route)**
```json
{
  "nodes": [
    {"id": "classify", "type": "llm", ...},
    {"id": "route", "type": "if", "condition": "${classify.category == 'urgent'}"},
    {"id": "urgent_path", "type": "tool", ...},
    {"id": "normal_path", "type": "tool", ...}
  ],
  "edges": [
    {"source": "classify", "target": "route"},
    {"source": "route", "target": "urgent_path", "type": "conditional_true"},
    {"source": "route", "target": "normal_path", "type": "conditional_false"}
  ]
}
```

**Pattern 3: Tool → ForEach → Tool (Fetch then Process Each)**
```json
{
  "nodes": [
    {"id": "fetch_list", "type": "tool", "tool": "list_items", "inputs": {}},
    {"id": "loop", "type": "for_each", "items": "${fetch_list.items}"},
    {"id": "process", "type": "tool", "tool": "update_item", "inputs": {"id": "${item.id}", "status": "processed"}}
  ],
  "edges": [
    {"source": "fetch_list", "target": "loop"},
    {"source": "loop", "target": "process", "type": "loop_body"},
    {"source": "process", "target": "loop"},
    {"source": "loop", "target": "done", "type": "loop_exit"}
  ]
}
```

---

## Quick Reference Table

| Block Type | Purpose | Required Fields | Output Access |
|------------|---------|----------------|---------------|
| `tool` | Execute registry tool | `id`, `tool`, `inputs` | `${node_id}`, `${node_id.field}` |
| `llm` | AI inference | `id`, `inputs.model`, `inputs.prompt`, `outputs` | `${node_id}`, `${node_id.field}` |
| `mcp` | External MCP tool | `id`, `server`, `tool` | `${node_id}`, `${node_id.field}` |
| `if` | Conditional branch | `id`, `condition` | N/A (routing only) |
| `for_each` | Loop over list | `id`, `items` | Loop state (internal) |

---
"""
    return guide


@lru_cache(maxsize=1)
def generate_graph_structure_guide() -> str:
    """
    Generate comprehensive guidance about workflow graph structure and compilation.

    Provides detailed documentation about:
    - How nodes and edges compile to LangGraph
    - Multiple edges handling (incoming/outgoing)
    - Diamond patterns and merging
    - Entry and exit points
    - Control flow routing
    - State management
    - Key constraints and rules

    Returns:
        Formatted markdown guide for graph structure
    """
    guide = """# Workflow Graph Structure & Compilation Guide

## Overview

Seer workflows are directed graphs compiled to LangGraph StateGraphs. Understanding how the compiler transforms your nodes and edges into executable graphs is crucial for building correct workflows.

---

## Compilation Pipeline

The compiler transforms WorkflowSpec JSON through these stages:

1. **Parse** - Validate JSON against Pydantic schema
2. **Build Type Environment** - Infer node output types from tools/models
3. **Validate References** - Check all `${...}` expressions resolve
4. **Lower Control Flow** - Build ExecutionPlan with edge indices
5. **Emit LangGraph** - Generate executable StateGraph

---

## Graph Structure Fundamentals

### Nodes
- Each node becomes a LangGraph node with a unique ID
- Nodes are functions that transform state: `(state, config) -> state_update`
- All node outputs are stored in state under their node ID
- State updates are **merged**, not replaced (preserves all node outputs)

### Edges
- Define execution flow between nodes
- Can be **direct** (sequential) or **conditional** (routing)
- Multiple edges FROM a node → conditional routing
- Multiple edges TO a node → merge point (both paths can reach it)

### State Management
- State is a dictionary: `{node_id: output, ...}`
- State updates use a **merge reducer**: `{**left, **right}`
- All node outputs persist throughout execution
- Trace keys (`_trace_*`) track execution metadata

---

## Entry Points (START Routing)

### Single Entry Point
When workflow has no triggers, START connects to the first node:

```json
{
  "nodes": [{"id": "first", ...}],
  "edges": []
}
```

**Compiled:** `START → first`

### Trigger-Based Entry
When workflow has triggers, START routes through bootstrap node:

```json
{
  "triggers": [
    {"id": "trigger1", ...},
    {"id": "trigger2", ...}
  ],
  "edges": [
    {"source": "trigger1", "target": "node_a", "type": "trigger"},
    {"source": "trigger2", "target": "node_b", "type": "trigger"}
  ]
}
```

**Compiled:**
- `START → __trigger_bootstrap`
- `__trigger_bootstrap` reads `_trigger_id` from state
- Routes to appropriate target based on which trigger fired

---

## Exit Points (END Routing)

### Explicit END
Nodes with no outgoing edges automatically connect to END:

```json
{
  "nodes": [
    {"id": "task", ...}
  ],
  "edges": []
}
```

**Compiled:** `task → END`

### Conditional END
Control flow nodes can route to END if no target specified:

```json
{
  "id": "check",
  "type": "if",
  "condition": "${done}"
}
```

**If only one branch has a target, other routes to END**

---

## Edge Types & Routing

### Default Edges (Sequential Flow)
```json
{"source": "node_a", "target": "node_b"}
{"source": "node_a", "target": "node_b", "type": "default"}
```

**Behavior:** Direct edge, node_b executes after node_a

### Multiple Outgoing Edges
**NOT ALLOWED on regular nodes.** Only control flow nodes (if, for_each) can have multiple outgoing edges.

❌ **Invalid:**
```json
{
  "nodes": [{"id": "task", "type": "tool", ...}],
  "edges": [
    {"source": "task", "target": "a"},
    {"source": "task", "target": "b"}
  ]
}
```

✅ **Valid (use if node):**
```json
{
  "nodes": [
    {"id": "task", "type": "tool", ...},
    {"id": "route", "type": "if", "condition": "${task.result == 'A'}"},
    {"id": "a", "type": "tool", ...},
    {"id": "b", "type": "tool", ...}
  ],
  "edges": [
    {"source": "task", "target": "route"},
    {"source": "route", "target": "a", "type": "conditional_true"},
    {"source": "route", "target": "b", "type": "conditional_false"}
  ]
}
```

### Conditional Edges (If Node)
If nodes REQUIRE exactly two edges:

```json
{
  "source": "if_node",
  "target": "true_path",
  "type": "conditional_true"
},
{
  "source": "if_node",
  "target": "false_path",
  "type": "conditional_false"
}
```

**Compiled:**
- If node evaluates condition, stores result in `_if_result_{node_id}`
- Router function reads result and routes to appropriate target
- LangGraph uses `add_conditional_edges(node_id, router, path_map)`

### Loop Edges (ForEach Node)
ForEach nodes REQUIRE exactly two edges:

```json
{
  "source": "loop_node",
  "target": "body_node",
  "type": "loop_body"
},
{
  "source": "loop_node",
  "target": "exit_node",
  "type": "loop_exit"
}
```

**Compiled:**
- Loop node manages iteration state in `_loop_{node_id}`
- Router checks `has_more_iterations` to decide loop_body vs loop_exit
- Terminal nodes in loop body automatically get implicit edge back to loop node

### Trigger Edges
```json
{
  "source": "trigger_id",
  "target": "first_node",
  "type": "trigger"
}
```

**Behavior:** Routes from trigger bootstrap to first processing node

---

## Diamond Patterns & Merging

### Diamond Pattern (If/Else Merge)
Two paths that merge back to a single node:

```json
{
  "nodes": [
    {"id": "start", "type": "tool", ...},
    {"id": "check", "type": "if", "condition": "${start.value > 0}"},
    {"id": "path_a", "type": "tool", ...},
    {"id": "path_b", "type": "tool", ...},
    {"id": "merge", "type": "tool", ...}
  ],
  "edges": [
    {"source": "start", "target": "check"},
    {"source": "check", "target": "path_a", "type": "conditional_true"},
    {"source": "check", "target": "path_b", "type": "conditional_false"},
    {"source": "path_a", "target": "merge"},
    {"source": "path_b", "target": "merge"}
  ]
}
```

**Behavior:**
- Both `path_a` and `path_b` have edges to `merge`
- LangGraph handles multiple incoming edges naturally
- `merge` node executes after either path completes
- State contains outputs from: `start`, `check`, (either `path_a` OR `path_b`), `merge`

**Important:** The merge node can reference outputs from either path using if/else logic:
```json
{
  "id": "merge",
  "type": "llm",
  "inputs": {
    "model": "gpt-5-mini",
    "prompt": "Result from path A: ${path_a}, Result from path B: ${path_b}"
  }
}
```

**Note:** Only ONE path executes, so only one of `${path_a}` or `${path_b}` will have a value.

---

## Multiple Incoming Edges

Multiple nodes can target the same node (merge point):

```json
{
  "edges": [
    {"source": "node_a", "target": "merge"},
    {"source": "node_b", "target": "merge"},
    {"source": "node_c", "target": "merge"}
  ]
}
```

**Valid patterns:**
1. **Sequential merge:** Different sequential paths lead to same node (OK but unusual)
2. **Conditional merge:** If/else branches merge (diamond pattern - common)
3. **Loop merge:** Loop exit leads to node that other paths also reach

**LangGraph behavior:** Node executes when ANY incoming edge is followed.

---

## Loop Body Detection

The compiler automatically detects loop body nodes:

```json
{
  "nodes": [
    {"id": "loop", "type": "for_each", "items": "${items}"},
    {"id": "process", "type": "tool", ...},
    {"id": "transform", "type": "tool", ...}
  ],
  "edges": [
    {"source": "loop", "target": "process", "type": "loop_body"},
    {"source": "process", "target": "transform"},
    {"source": "transform", "target": "loop"},
    {"source": "loop", "target": "done", "type": "loop_exit"}
  ]
}
```

**Compilation:**
1. Compiler finds `loop_body` edge: `loop → process`
2. Follows `default` edges from `process`: `process → transform`
3. Detects `transform` has edge back to `loop`
4. Loop body nodes: `{process, transform}`
5. Terminal nodes: `{transform}` (already has edge back to loop)

**Implicit loop back edges:**
If terminal node doesn't have explicit edge to loop, compiler adds one:

```json
{
  "edges": [
    {"source": "loop", "target": "process", "type": "loop_body"},
    {"source": "process", "target": "transform"}
    // No edge from transform back to loop
  ]
}
```

**Compiled:** Implicit edge added: `transform → loop`

---

## State Merging & Trace Keys

### State Updates
Every node execution produces a state update dict:
```python
{
  "node_id": <node_output>,
  "_trace_node_id": {"inputs": ..., "output": ..., "timestamp": ...}
}
```

### Merge Strategy
State updates use **merge reducer** (not replace):
```python
def merge_state(left: dict, right: dict) -> dict:
    return {**left, **right}
```

**Example:**
```
Initial state: {}
After node_a: {"node_a": "result_a", "_trace_node_a": {...}}
After node_b: {"node_a": "result_a", "_trace_node_a": {...}, "node_b": "result_b", "_trace_node_b": {...}}
```

**Important:** All node outputs persist throughout execution. You can reference earlier nodes from later nodes.

---

## Key Constraints & Rules

### Node Constraints
1. ✅ **Unique IDs:** Every node must have a unique `id`
2. ✅ **Reachability:** All nodes must be reachable from START (no orphaned nodes)
3. ❌ **No Self-Edges:** Node cannot connect to itself (except through loop pattern)
4. ❌ **No Multiple Outgoing Edges:** Only if/for_each nodes can have conditional edges

### Edge Constraints
1. ✅ **Valid Source/Target:** Both must exist in nodes list
2. ✅ **If Node Edges:** Must have exactly one `conditional_true` and one `conditional_false`
3. ✅ **ForEach Node Edges:** Must have exactly one `loop_body` and one `loop_exit`
4. ✅ **Trigger Edges:** Source is trigger ID, target is node ID
5. ❌ **Circular Dependencies:** Avoid cycles (except loop pattern)

### Control Flow Constraints
1. **If Node:**
   - Must have `condition` field
   - Must have both conditional edges
   - Cannot have default edges

2. **ForEach Node:**
   - Must have `items` field
   - Must have both loop edges
   - Loop body nodes automatically detected
   - Terminal nodes get implicit back-edge

3. **Tool/LLM/MCP Nodes:**
   - Can only have default edges
   - Cannot have conditional edges
   - Can have zero or more outgoing edges (if zero, routes to END)

---

## Common Patterns

### Pattern 1: Linear Flow
```
START → node_a → node_b → node_c → END
```

### Pattern 2: Conditional Branch
```
START → check_node (if) → path_a → END
                        └→ path_b → END
```

### Pattern 3: Diamond (Branch + Merge)
```
START → node_a → check (if) → path_a → merge → END
                            └→ path_b ┘
```

### Pattern 4: Loop
```
START → loop (for_each) → process → transform → loop → END
         └──────────────────────────────────────┘
```

### Pattern 5: Nested Control Flow
```
START → outer_loop (for_each) → check (if) → path_a → outer_loop → END
                                           └→ path_b ┘
```

### Pattern 6: Multiple Triggers
```
START → __trigger_bootstrap → node_a (trigger1)
                            └→ node_b (trigger2)
                            └→ node_c (trigger3)
```

---

## Debugging Graph Issues

### Issue: "Node X is not reachable"
**Cause:** No path from START to node X
**Fix:** Add edges connecting START (or trigger) to the orphaned node

### Issue: "If node missing conditional edges"
**Cause:** If node doesn't have both `conditional_true` and `conditional_false` edges
**Fix:** Add both edge types

### Issue: "Multiple outgoing edges from tool node"
**Cause:** Tool/LLM/MCP node has more than one outgoing edge
**Fix:** Use If node for branching, not multiple edges from tool node

### Issue: "Circular reference detected"
**Cause:** Node A references node B, which references node A
**Fix:** Ensure dependencies flow in one direction (except in loop patterns)

### Issue: "ForEach loop body not detected"
**Cause:** No `loop_body` edge or broken edge chain in loop
**Fix:** Ensure loop has `loop_body` edge and body nodes connect back to loop

---

## Best Practices

1. **Keep graphs acyclic** (except loops)
2. **Use If nodes for branching**, not multiple edges from regular nodes
3. **Merge diamond patterns explicitly** with a merge node
4. **Name nodes descriptively** for easier debugging
5. **Validate edge types** match node types (conditional for if, loop for for_each)
6. **Test diamond patterns** to ensure both paths work correctly
7. **Use trigger routing** for event-driven workflows
8. **Avoid deep nesting** of control flow (keep it simple)

---

## Quick Reference

| Pattern | Structure | Use Case |
|---------|-----------|----------|
| Linear | A → B → C | Sequential operations |
| Branch | A → if → B or C | Route based on condition |
| Diamond | A → if → B/C → D | Branch then merge |
| Loop | for_each → body → for_each | Iterate over items |
| Multi-trigger | START → bootstrap → A/B/C | Event-driven routing |

---
"""
    return guide
