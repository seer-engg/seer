# Primitive Workflow Blocks Reference

## Overview
Seer workflows are built from 8 primitive block types: **tool**, **llm**, **mcp**, **if**, **for_each**, **hitl**, **image_gen**, and **browser**.
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

**Best Practice:** Design your data structures so items are objects with named properties (`{name: "...", email: "..."}`) rather than arrays requiring index access (`["...", "..."]`).

**Edge Configuration:**
ForEach nodes use two edge types:
- `loop_body`: Points to the first node in the loop body
- `loop_exit`: Points to the node executed after all iterations complete

```json
{
  "source": "loop_id",
  "target": "process_item",
  "type": "loop_body"
},
{
  "source": "loop_id",
  "target": "next_node",
  "type": "loop_exit"
}
```

**Note:** Back-edges from the last node in the loop body to the for_each node are **optional**. The compiler automatically adds implicit back-edges from terminal nodes (nodes with no outgoing edges in the loop body) back to the loop. You can add explicit back-edges if you prefer, but they are not required.

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

## 6. HITL BLOCK (`type: "hitl"`)

**Purpose:** Human-In-The-Loop - pause workflow execution to collect user input via web form or email

**Schema:**
```json
{
  "id": "unique_node_id",
  "type": "hitl",
  "title": "Approval Required",
  "description": "Please review and approve this action",
  "display": [
    {"label": "Item", "value": "${previous_node.name}"},
    {"label": "Amount", "value": "${previous_node.total}"}
  ],
  "inputs": [
    {
      "id": "decision",
      "question": "Do you approve this request?",
      "input_type": "single_choice",
      "options": [
        {"value": "approve", "label": "Approve"},
        {"value": "reject", "label": "Reject"}
      ],
      "required": true
    }
  ],
  "timeout_seconds": 86400,
  "delivery_channels": [
    {"type": "platform"}
  ]
}
```

**Required Fields:**
- `id` (string): Unique identifier
- `type`: Must be `"hitl"`
- `title` (string): Display title shown to the user

**Optional Fields:**
- `description` (string): Additional context text
- `display` (array): Items to show for context, each with `label` and `value` (supports `${...}` expressions)
- `inputs` (array): Input fields to collect from user
- `timeout_seconds` (number): Timeout in seconds (`null` or `0` = indefinite wait)
- `delivery_channels` (array): Notification channels (default: `platform` only)

**Input Types:**
| Type | Description | Requires Options |
|------|-------------|------------------|
| `single_choice` | User selects ONE option | Yes (min 2) |
| `multi_choice` | User selects MULTIPLE options | Yes (min 2) |
| `text` | Free-form text input | No |
| `number` | Numeric input | No |
| `boolean` | Yes/No toggle | No |

**Input Field Schema:**
```json
{
  "id": "field_id",
  "question": "What is your decision?",
  "input_type": "single_choice",
  "options": [
    {"value": "yes", "label": "Yes"},
    {"value": "no", "label": "No", "requires_text": true}
  ],
  "required": true,
  "placeholder": "Select an option",
  "default_value": "yes"
}
```

**Delivery Channels:**
- `platform`: Default web-based form (user polls `/runs/{id}/interrupt`)
- `gmail`: Email notification with form link

**Gmail Delivery Example:**
```json
{
  "delivery_channels": [
    {"type": "platform"},
    {
      "type": "gmail",
      "gmail": {
        "to": ["approver@example.com"],
        "subject": "Approval Required: ${item.name}"
      }
    }
  ]
}
```

**Important Notes:**
- ⏸️ Workflow execution **pauses** until user responds or timeout occurs
- ✅ User responses accessible via `${hitl_node_id.field_id}` in downstream nodes
- ✅ Use `display` items to show context data from previous nodes
- ✅ Multiple delivery channels can be combined (e.g., platform + email)

**Example - Data Review Workflow:**
```json
{
  "id": "review_data",
  "type": "hitl",
  "title": "Review Extracted Data",
  "description": "Please verify the extracted information is correct",
  "display": [
    {"label": "Customer Name", "value": "${extract.customer_name}"},
    {"label": "Order Total", "value": "${extract.total}"},
    {"label": "Items", "value": "${extract.item_count} items"}
  ],
  "inputs": [
    {
      "id": "is_correct",
      "question": "Is this data correct?",
      "input_type": "boolean",
      "required": true
    },
    {
      "id": "corrections",
      "question": "If incorrect, what needs to be fixed?",
      "input_type": "text",
      "required": false,
      "placeholder": "Describe any corrections needed"
    }
  ],
  "timeout_seconds": 3600
}
```

**Common Use Cases:**
- Approval workflows (expense, content, access requests)
- Data verification/correction
- Manual decision points
- Quality assurance checkpoints

---

## 7. IMAGE_GEN BLOCK (`type: "image_gen"`)

**Purpose:** Generate images using AI models via OpenRouter API

**Schema:**
```json
{
  "id": "unique_node_id",
  "type": "image_gen",
  "inputs": {
    "model": "openai/dall-e-3",
    "prompt": "A professional product photo of ${product.name}",
    "size": "1024x1024",
    "num_images": 1
  }
}
```

**Required Fields:**
- `id` (string): Unique identifier
- `type`: Must be `"image_gen"`
- `inputs` (object): Must contain:
  - `model` (string): Image generation model ID
  - `prompt` (string): Image description (supports `${...}` expressions)

**Optional Input Fields:**
- `size` (string): Image dimensions (e.g., `"1024x1024"`, `"1792x1024"`)
- `num_images` (number): Number of images to generate

**Available Models (via OpenRouter):**
- `openai/dall-e-3` - DALL-E 3
- `openai/dall-e-2` - DALL-E 2
- Other models supported by OpenRouter

**Important Notes:**
- ✅ Output contains generated image URL(s)
- ✅ Use descriptive prompts for better results
- ✅ Prompt supports `${...}` expressions for dynamic content

**Example:**
```json
{
  "id": "generate_thumbnail",
  "type": "image_gen",
  "inputs": {
    "model": "openai/dall-e-3",
    "prompt": "Create a minimalist thumbnail image for a blog post about: ${article.title}. Style: modern, clean, professional",
    "size": "1024x1024"
  }
}
```

**Common Use Cases:**
- Product image generation
- Social media graphics
- Blog post thumbnails
- Marketing visuals

---

## 8. BROWSER BLOCK (`type: "browser"`)

**Purpose:** Browser automation using natural language task descriptions

**Schema:**
```json
{
  "id": "unique_node_id",
  "type": "browser",
  "task": "Go to the website and extract the pricing information",
  "inputs": {
    "url": "${config.target_url}",
    "search_term": "${query}"
  },
  "browser_profile_id": "uuid-of-saved-profile",
  "max_steps": 25,
  "timeout_seconds": 300,
  "expect_outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "prices": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["prices"]
      }
    }
  },
  "save_screenshots": false
}
```

**Required Fields:**
- `id` (string): Unique identifier
- `type`: Must be `"browser"`
- `task` (string): Natural language description of what to do (supports `${...}` expressions)

**Optional Fields:**
- `inputs` (object): Additional context data passed to the browser agent
- `browser_profile_id` (string): Reference to saved browser profile with login sessions
- `max_steps` (number): Maximum automation steps (default: 25, range: 1-100)
- `timeout_seconds` (number): Execution timeout (default: 300, range: 30-1800)
- `expect_outputs` (object): Output schema for structured data extraction
- `save_screenshots` (boolean): Save screenshots to S3 (default: false)

**Browser Profiles:**
Browser profiles contain saved login sessions, allowing workflows to interact with authenticated websites (Slack, email, dashboards, etc.) without storing credentials in the workflow.

- Profiles are managed separately via `/browser/profiles` API endpoints
- Reference a profile by its UUID in `browser_profile_id`
- The browser agent loads the profile's cookies/session state before executing the task

**Important Notes:**
- 🤖 Uses BrowserUse Agent for intelligent browser automation
- ✅ Natural language tasks - describe WHAT you want, not HOW
- ✅ `inputs` provide additional context data the agent can reference
- ✅ Use `expect_outputs` for structured data extraction
- ⏱️ Default timeout is 5 minutes; increase for complex tasks

**Example - Web Scraping:**
```json
{
  "id": "scrape_prices",
  "type": "browser",
  "task": "Navigate to ${inputs.url}, find the pricing table, and extract all plan names and their monthly prices",
  "inputs": {
    "url": "https://example.com/pricing"
  },
  "max_steps": 30,
  "timeout_seconds": 120,
  "expect_outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "plans": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {"type": "string"},
                "price": {"type": "string"}
              }
            }
          }
        },
        "required": ["plans"]
      }
    }
  }
}
```

**Example - Authenticated Task:**
```json
{
  "id": "get_slack_messages",
  "type": "browser",
  "task": "Go to Slack, navigate to the #general channel, and get the last 5 messages",
  "browser_profile_id": "abc123-profile-uuid",
  "max_steps": 20,
  "expect_outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "messages": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "author": {"type": "string"},
                "text": {"type": "string"},
                "timestamp": {"type": "string"}
              }
            }
          }
        },
        "required": ["messages"]
      }
    }
  }
}
```

**Common Use Cases:**
- Web scraping and data extraction
- Form filling and submissions
- Authenticated website interactions
- Screenshot capture for documentation
- E-commerce monitoring

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
| `hitl` | Human-In-The-Loop | `id`, `title` | `${node_id.input_field_id}` |
| `image_gen` | Generate images | `id`, `inputs.model`, `inputs.prompt` | `${node_id}` (image URL) |
| `browser` | Browser automation | `id`, `task` | `${node_id}`, `${node_id.field}` |

---
