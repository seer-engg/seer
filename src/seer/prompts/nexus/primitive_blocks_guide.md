# Primitive Workflow Blocks Reference

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

**⚠️ CRITICAL: Root Schema Must Be `object`**
OpenAI structured outputs require the root schema to have `"type": "object"`. **Array root types are NOT supported** and will fail at compile time.

❌ **WRONG** - Array at root:
```json
{
  "outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "array",
        "items": {"type": "object", "properties": {"name": {"type": "string"}}}
      }
    }
  }
}
```

✅ **CORRECT** - Wrap array in object property:
```json
{
  "outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "items": {
            "type": "array",
            "items": {"type": "object", "properties": {"name": {"type": "string"}}}
          }
        },
        "required": ["items"]
      }
    }
  }
}
```
Then access the array via `${node_id.items}` in downstream nodes.

**Example:**
```json
{
  "id": "classify_email",
  "type": "llm",
  "inputs": {
    "model": "gpt-5-mini",
    "temperature": 0.3,
    "prompt": "Classify this email:\n\nFrom: ${email.from}\nSubject: ${email.subject}\nBody: ${email.body}",
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

**⚠️ Type Inference Limitation:**
Loop variables (`item_var`) receive a permissive object schema at compile time because the actual item type cannot always be inferred. This means:
- ✅ Object property access works: `${item.name}`, `${item.email}`
- ⚠️ Array indexing may fail: `${item[0]}` - only works if the compiler knows `item` is an array type

**If iterating over array-of-arrays (e.g., spreadsheet rows):**
Each `item` is `["col1", "col2", "col3"]` but may be typed as object, not array.

**Recommended workaround - use LLM to extract values:**
```json
{
  "id": "parse_row",
  "type": "llm",
  "inputs": {
    "model": "gpt-5-mini",
    "prompt": "Extract from this spreadsheet row: ${item}. Expected columns: Name (index 0), Email (index 1), Status (index 2). Return as JSON with name, email, status fields."
  },
  "outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "email": {"type": "string"},
          "status": {"type": "string"}
        },
        "required": ["name", "email", "status"]
      }
    }
  }
}
```

**Best Practice:** Design your data structures so items are objects with named properties (`{name: "...", email: "..."}`) rather than arrays requiring index access (`["...", "..."]`).

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
    },
    {
      "id": "notify_complete",
      "type": "tool",
      "tool": "gmail_send_email",
      "inputs": {
        "to": ["admin@example.com"],
        "subject": "Batch emails complete",
        "body": "All notification emails have been sent."
      }
    }
  ],
  "edges": [
    {"source": "loop_emails", "target": "send_email", "type": "loop_body"},
    {"source": "send_email", "target": "loop_emails", "type": "default"},
    {"source": "loop_emails", "target": "notify_complete", "type": "loop_exit"}
  ]
}
```

**Important:** The `loop_exit` target MUST be an existing node ID (like `notify_complete` above). Never use undefined targets like `"done"` - this will cause validation errors.

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

## Expression Limitations (IMPORTANT)

**Template expressions and condition expressions have DIFFERENT capabilities.**

### Template Expressions (in tool inputs, LLM prompts)
Template expressions perform simple substitution only - they resolve references and convert to strings.

**✅ Supported:**
- Simple references: `${node_id}`, `${node_id.field}`, `${node.nested.field}`
- Array indexing: `${array[0]}`, `${data[0].field}` (if type is known)
- Object key access: `${obj['key-with-dashes']}`

**❌ NOT Supported in Templates:**
- Arithmetic: `${index + 1}` - **WILL FAIL**
- String concatenation: `${first + " " + last}` - **WILL FAIL**
- Function calls: `${len(items)}` - **WILL FAIL**
- Comparisons: `${count > 0}` - **WILL FAIL**

**Example - WRONG:**
```json
{
  "inputs": {
    "row_number": "${index + 2}"
  }
}
```
This will fail because arithmetic is not allowed in template expressions.

**Example - CORRECT (workaround using LLM):**
```json
{
  "id": "compute_row",
  "type": "llm",
  "inputs": {
    "model": "gpt-5-mini",
    "prompt": "Calculate and return only the number: ${index} + 2"
  },
  "outputs": {"mode": "text"}
}
```
Then use `${compute_row}` in subsequent nodes.

### Condition Expressions (in `if` node conditions)
Conditions are evaluated with full Python-like expression support.

**✅ Supported:**
- Arithmetic: `${count} + 1 > 5`, `${price} * ${quantity} > 1000`
- Comparisons: `${status} == 'active'`, `${count} >= 10`
- Boolean logic: `${flag} && ${other}`, `!${disabled}`, `${a} || ${b}`
- Functions: `len(${items}) > 0`, `sum(${values}) < 100`, `any(${flags})`

**Available Functions:** `len`, `any`, `all`, `min`, `max`, `sum`, `str`, `int`, `float`

**Example - CORRECT:**
```json
{
  "id": "check_threshold",
  "type": "if",
  "condition": "${count} + ${offset} > 10"
}
```
Arithmetic works in `if` conditions!

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
    {"id": "process", "type": "tool", "tool": "update_item", "inputs": {"id": "${item.id}", "status": "processed"}},
    {"id": "complete", "type": "llm", "inputs": {"model": "gpt-5-mini", "prompt": "Summarize: processed all items"}, "outputs": {"mode": "text"}}
  ],
  "edges": [
    {"source": "fetch_list", "target": "loop"},
    {"source": "loop", "target": "process", "type": "loop_body"},
    {"source": "process", "target": "loop"},
    {"source": "loop", "target": "complete", "type": "loop_exit"}
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
