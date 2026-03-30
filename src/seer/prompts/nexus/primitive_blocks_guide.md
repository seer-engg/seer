# Primitive Workflow Blocks Reference

## Overview
Seer workflows are built from 8 primitive block types: **tool**, **agent**, **mcp**, **if**, **for_each**, **hitl**, **image_gen**, and **browser**.
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
    "to": ["${email_trigger.data.recipient}"],
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

## 2. MCP BLOCK (`type: "mcp"`)

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
- `expect_outputs` (object): Output validation schema (same as agent outputs)

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

## 3. IF BLOCK (`type: "if"`)

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

## 4. FOR_EACH BLOCK (`type: "for_each"`)

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
- `items` (string): Expression that resolves to a list (e.g., `"${results}"`, `"${my_trigger.data.items}"`)

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

## 5. HITL BLOCK (`type: "hitl"`)

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
| `table` | Batch review with per-row inputs | No (uses `columns`) |

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

**Table Input Field Schema:**
```json
{
  "id": "review_items",
  "question": "Review each item below:",
  "input_type": "table",
  "row_data_expression": "${fetch_items.body.results}",
  "row_display_fields": [
    {"label": "Title", "value": "${row.title}"},
    {"label": "Author", "value": "${row.author}"}
  ],
  "columns": [
    {
      "id": "rating",
      "header": "Rating",
      "input_type": "single_choice",
      "options": [
        {"value": "good", "label": "Good"},
        {"value": "bad", "label": "Bad"}
      ]
    },
    {
      "id": "notes",
      "header": "Notes",
      "input_type": "text",
      "required": false
    }
  ],
  "required": true
}
```
- `row_data_expression`: Expression resolving to an array from workflow state (e.g., HTTP response body)
- `row_display_fields`: Read-only columns shown per row, evaluated with `${row.*}` context
- `columns`: Editable input columns per row (same types as regular inputs, except no nested `table`)
- Response shape: `{"review_items": [{"rating": "good", "notes": "..."}, ...]}` — one object per row

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

**Example - Batch Review from API:**
```json
{
  "id": "review_highlights",
  "type": "hitl",
  "title": "Review Readwise Highlights",
  "description": "Rate each highlight and add notes",
  "inputs": [
    {
      "id": "reviews",
      "question": "Review each highlight:",
      "input_type": "table",
      "row_data_expression": "${fetch_highlights.body.results}",
      "row_display_fields": [
        {"label": "Text", "value": "${row.text}"},
        {"label": "Book", "value": "${row.book_title}"}
      ],
      "columns": [
        {
          "id": "action",
          "header": "Action",
          "input_type": "single_choice",
          "options": [
            {"value": "keep", "label": "Keep"},
            {"value": "discard", "label": "Discard"}
          ]
        },
        {
          "id": "tags",
          "header": "Tags",
          "input_type": "text",
          "required": false
        }
      ]
    }
  ]
}
```

**Common Use Cases:**
- Approval workflows (expense, content, access requests)
- Data verification/correction
- Manual decision points
- Quality assurance checkpoints
- Batch review of items from API responses (e.g., highlights, tickets, records)

---

## 6. IMAGE_GEN BLOCK (`type: "image_gen"`)

**Purpose:** Generate images using AI models via OpenRouter API

**Schema:**
```json
{
  "id": "unique_node_id",
  "type": "image_gen",
  "inputs": {
    "model": "sourceful/riverflow-v2-fast",
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

**Available Models:**
- `sourceful/riverflow-v2-fast` - Fast image generation
- `google/gemini-2.5-flash-image` - Gemini image generation

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
    "model": "sourceful/riverflow-v2-fast",
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

## 7. BROWSER BLOCK (`type: "browser"`)

**Purpose:** Browser automation using natural language task descriptions powered by an LLM-driven agent.

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
- `model` (string): OpenRouter model ID for browser agent (default: `qwen/qwen3-vl-8b-thinking`)
- `max_steps` (number): Maximum automation steps (default: 25, range: 1-100)
- `timeout_seconds` (number): Execution timeout (default: 300, range: 30-1800)
- `expect_outputs` (object): Output schema for structured data extraction
- `save_screenshots` (boolean): Save screenshots to S3 (default: false)

**Browser Profiles:**
Browser profiles contain saved login sessions, allowing workflows to interact with authenticated websites (Slack, email, dashboards, etc.) without storing credentials in the workflow.

- Profiles are managed separately via `/browser/profiles` API endpoints
- Reference a profile by its UUID in `browser_profile_id`
- The browser agent loads the profile's cookies/session state before executing the task

---

### Output Format

Browser nodes always return an envelope structure with these fields:

```json
{
  "success": true,
  "result": "Step-by-step history of actions taken",
  "extracted_data": { "field": "value" },
  "final_url": "https://example.com/page",
  "screenshots": [
    { "id": "uuid", "url": "https://...", "name": "step_1.png" }
  ],
  "usage": {
    "input_tokens": 1500,
    "output_tokens": 200,
    "steps_taken": 8
  }
}
```

**Critical Reference Pattern:**
```
✅ Correct: ${browser_node.extracted_data.field}
✅ Correct: ${browser_node.success}
✅ Correct: ${browser_node.screenshots}

❌ Wrong: ${browser_node.field}  (fields are inside extracted_data!)
```

---

### Prompting Best Practices

Effective prompting dramatically improves browser automation reliability.

**1. Be Specific, Not Open-Ended**

✅ **Good - Specific steps:**
```
1. Go to https://quotes.toscrape.com/
2. Use extract action to get the first 3 quotes with their authors
3. Return the results as structured data
```

❌ **Bad - Vague:**
```
Go to the web and find some quotes
```

**2. Name Actions Directly**

Reference browser actions by name for predictable behavior:

| Action | Description | Example Usage |
|--------|-------------|---------------|
| `go_to_url` | Navigate to a URL | "Use go_to_url to navigate to..." |
| `click` | Click an element | "Click the 'Submit' button" |
| `type` / `input_text` | Enter text into a field | "Type 'search term' into the search box" |
| `scroll` | Scroll the page | "Scroll down 2 pages" |
| `extract` | Extract content | "Use extract action with query 'product prices'" |
| `send_keys` | Send keyboard input | "Use send_keys with 'Tab Tab Enter'" |
| `go_back` | Navigate back | "Use go_back to return to previous page" |
| `wait` | Wait for element/load | "Wait for the results to load" |

**3. Number Your Steps for Complex Tasks**

```
1. Navigate to https://example.com/login
2. Type "${inputs.username}" into the username field
3. Type "${inputs.password}" into the password field
4. Click the "Sign In" button
5. Wait for the dashboard to load
6. Use extract action to get account balance
```

**4. Handle Interaction Problems via Keyboard**

Sometimes buttons can't be clicked. Use keyboard navigation as fallback:

```
If the submit button cannot be clicked:
1. Use send_keys action with "Tab Tab Enter" to navigate and activate
2. Or use send_keys with "ArrowDown ArrowDown Enter" for dropdown selection
```

---

### Error Recovery Patterns

Include fallback strategies in complex tasks:

```
Login and extract account data:
1. Navigate to dashboard.example.com
2. If login page appears, enter credentials and submit
3. If navigation fails due to anti-bot protection:
   - Use google search to find "example.com dashboard login"
   - Navigate via search results
4. If page times out:
   - Use go_back and try alternative approach
   - Or refresh the page and wait 5 seconds
5. Extract the account balance and recent transactions
```

---

### Cost & Performance Tips

- Browser tasks use multiple LLM calls (one per decision step) - they can be expensive
- `max_steps` affects both completion probability AND cost - start with defaults
- Default timeout (5 min) is sufficient for most tasks; increase only for complex multi-page workflows
- Enable `save_screenshots` only when debugging or documentation is needed
- For simple extractions, consider using `tool` blocks with scraping tools instead

---

### Examples

**Example 1 - Web Scraping with Specific Actions:**
```json
{
  "id": "scrape_prices",
  "type": "browser",
  "task": "1. Navigate to ${inputs.url}\n2. Scroll down to find the pricing table\n3. Use extract action to get all plan names and monthly prices\n4. Return structured data with plans array",
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

**Example 2 - Authenticated Task with Profile:**
```json
{
  "id": "get_slack_messages",
  "type": "browser",
  "task": "1. Go to Slack workspace\n2. Navigate to the #general channel using the sidebar\n3. Scroll up to load message history if needed\n4. Use extract action to get the last 5 messages with author, text, and timestamp",
  "browser_profile_id": "abc123-profile-uuid",
  "max_steps": 25,
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

**Example 3 - Form Submission with Keyboard Fallback:**
```json
{
  "id": "submit_contact_form",
  "type": "browser",
  "task": "1. Navigate to ${inputs.form_url}\n2. Type '${inputs.name}' into the Name field\n3. Type '${inputs.email}' into the Email field\n4. Type '${inputs.message}' into the Message textarea\n5. Click the Submit button\n6. If the button cannot be clicked, use send_keys with 'Tab Enter'\n7. Wait for confirmation message\n8. Extract the confirmation text or reference number",
  "inputs": {
    "form_url": "https://example.com/contact",
    "name": "${form_trigger.data.sender_name}",
    "email": "${form_trigger.data.sender_email}",
    "message": "${form_trigger.data.message}"
  },
  "max_steps": 20,
  "timeout_seconds": 60
}
```

**Example 4 - Screenshot Documentation:**
```json
{
  "id": "capture_dashboard",
  "type": "browser",
  "task": "1. Navigate to the analytics dashboard\n2. Wait for all charts to fully load\n3. Scroll through the entire page to capture all sections\n4. The screenshots will document the current state",
  "browser_profile_id": "analytics-profile-uuid",
  "save_screenshots": true,
  "max_steps": 15,
  "timeout_seconds": 90
}
```

**Example 5 - Error Recovery with Fallback:**
```json
{
  "id": "find_ceo_info",
  "type": "browser",
  "task": "1. Navigate to openai.com/about\n2. Find and extract the CEO's name and bio\n3. If navigation fails due to anti-bot protection:\n   - Use google search for 'OpenAI CEO name'\n   - Extract the information from search results\n4. If the about page doesn't have CEO info:\n   - Try navigating to openai.com/team\n   - Or search for 'OpenAI leadership team'",
  "max_steps": 40,
  "timeout_seconds": 180,
  "expect_outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "ceo_name": {"type": "string"},
          "bio": {"type": "string"},
          "source_url": {"type": "string"}
        },
        "required": ["ceo_name"]
      }
    }
  }
}
```

**Common Use Cases:**
- Web scraping and structured data extraction
- Form filling and automated submissions
- Authenticated website interactions (with browser profiles)
- Screenshot capture for documentation/monitoring
- E-commerce price monitoring
- Multi-step research with fallback strategies

---

## 8. AGENT BLOCK (`type: "agent"`)

**Purpose:** Multi-step autonomous task execution with tool access. The agent reasons, calls tools, and iterates until the task is complete.

**Schema:**
```json
{
  "id": "unique_node_id",
  "type": "agent",
  "inputs": {
    "model": "qwen/qwen3-235b-a22b-2507",
    "prompt": "Research ${topic} and compile a summary with key findings",
    "tools": ["web_search", "gmail_send_email"],
    "max_iterations": 10,
    "temperature": 0.3
  },
  "outputs": {
    "mode": "text"
  }
}
```

**Required Fields:**
- `id` (string): Unique identifier for this node
- `type`: Must be `"agent"`
- `inputs` (object): Must contain:
  - `model` (string): Model ID (e.g., `"qwen/qwen3-235b-a22b-2507"`, `"google/gemini-2.0-flash-001"`, `"mistralai/mistral-small-3.2-24b-instruct"`)
  - `prompt` (string): Task description with `${...}` expressions for dynamic content

**Optional Input Fields:**
- `tools` (array): List of tools the agent can call autonomously
  - Simple format: `["tool_name", "another_tool"]`
  - With OAuth: `[{"name": "gmail_send_email", "connection_id": 42}]`
- `max_iterations` (number): Maximum reasoning/tool-calling steps (optional, unlimited if not set)
- `temperature` (number): LLM temperature for agent reasoning (default: 0.2)

**Output Configuration:**
- `outputs` (object): Defines output format
  - `mode`: `"text"` (default) or `"json"`
  - `schema`: Required if `mode="json"`, contains JSON Schema

**Tool Format:**
Tools can be specified in two formats:

1. **Simple string** - tool name only:
```json
"tools": ["web_search", "supabase_select_rows", "gmail_get_message"]
```

2. **Object with connection_id** - for OAuth tools with multiple accounts:
```json
"tools": [
  "web_search",
  {"name": "gmail_send_email", "connection_id": 42},
  {"name": "slack_send_message", "connection_id": 15}
]
```

**Important Notes:**
- ✅ Use `search_tools(query)` to discover available tools and their exact names
- ✅ For OAuth tools (gmail, slack, google, etc.), call `get_tool_accounts(tool_name)` first
- ✅ If user has multiple OAuth accounts, include `connection_id` in the tool spec
- ✅ Agent output is accessed via `${node_id}` or `${node_id.field}` (if JSON mode)
- ⚠️ Agents can be expensive - each iteration involves LLM calls and potentially tool execution
- ⚠️ Use `max_iterations` to prevent runaway execution (recommended for production workflows)
- ✉️ **Email body generation:** When tasking an agent to generate an email body, instruct it to produce the content as **formatted Markdown** (use headings, bullet points, bold text, etc.). Email tools convert Markdown to HTML for rendering in email clients.

**Output Modes:**

1. **Text Mode** (default - freeform text):
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
          "summary": {"type": "string"},
          "findings": {
            "type": "array",
            "items": {"type": "string"}
          },
          "confidence": {"type": "number"}
        },
        "required": ["summary", "findings"]
      }
    }
  }
}
```

**Example 1 - Research Agent:**
```json
{
  "id": "research_agent",
  "type": "agent",
  "inputs": {
    "model": "qwen/qwen3-235b-a22b-2507",
    "prompt": "Research the company ${webhook_trigger.data.company_name}. Find their:\n1. Main products/services\n2. Recent news or announcements\n3. Key leadership\n\nCompile a brief executive summary.",
    "tools": ["web_search"],
    "max_iterations": 15
  },
  "outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "company": {"type": "string"},
          "products": {"type": "array", "items": {"type": "string"}},
          "recent_news": {"type": "array", "items": {"type": "string"}},
          "leadership": {"type": "array", "items": {"type": "string"}},
          "summary": {"type": "string"}
        },
        "required": ["company", "summary"]
      }
    }
  }
}
```

**Example 2 - Email Processing Agent with OAuth:**
```json
{
  "id": "email_processor",
  "type": "agent",
  "inputs": {
    "model": "qwen/qwen3-235b-a22b-2507",
    "prompt": "Process the email thread ${email_thread.id}:\n1. Read all messages in the thread\n2. Summarize the key points\n3. Draft a professional response addressing the main concerns\n4. Save the draft (do not send)",
    "tools": [
      {"name": "gmail_get_thread", "connection_id": 42},
      {"name": "gmail_create_draft", "connection_id": 42}
    ],
    "max_iterations": 10,
    "temperature": 0.3
  },
  "outputs": {"mode": "text"}
}
```

**Example 3 - Data Enrichment Agent:**
```json
{
  "id": "enrich_contact",
  "type": "agent",
  "inputs": {
    "model": "qwen/qwen3-235b-a22b-2507",
    "prompt": "Enrich the contact information for ${item.email}:\n1. Search for their LinkedIn profile\n2. Find their current company and title\n3. Look for recent professional activity\nReturn structured data.",
    "tools": ["web_search"],
    "max_iterations": 8
  },
  "outputs": {
    "mode": "json",
    "schema": {
      "json_schema": {
        "type": "object",
        "properties": {
          "email": {"type": "string"},
          "name": {"type": "string"},
          "company": {"type": "string"},
          "title": {"type": "string"},
          "linkedin_url": {"type": "string"}
        },
        "required": ["email"]
      }
    }
  }
}
```

**Common Use Cases:**
- Research and information gathering
- Multi-step data processing pipelines
- Automated email/message handling with context
- Data enrichment from multiple sources
- Complex decision-making with tool access
- Report generation with live data

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
- ⚠️ Only `http_request` tool outputs have a `.body` wrapper (e.g., `${fetch.body.items}`). Other tools return data directly — use `${node_id.field}` without `.body` (e.g., `${read_sheet.values}`, `${gmail.subject}`). Check `search_tools()` output schema for exact field names.

**Loop Variables:**
- `${item}` - Current item in for_each loop (or custom `item_var`)
- `${index}` - Current index in for_each loop (or custom `index_var`)

**Global Variables (Settings → Variables):**
- `${vars.KEY_NAME}` - Access organization-level variables (API keys, config values)
- Example: `${vars.READWISE_TOKEN}`, `${vars.SLACK_WEBHOOK_URL}`
- ⚠️ Use `${vars.*}` NOT `${variables.*}` or `${secrets.*}`

**Secrets (MCP only):**
- `${secrets.api_key}` - Reference to stored credentials

**Operators:**
- Comparison: `==`, `!=`, `>`, `<`, `>=`, `<=`
- Boolean: `&&` (and), `||` (or), `!` (not)
- Array access: `${array[0]}`, `${array[0].field}`

---

## Expression Limitations (IMPORTANT)

**Template expressions and condition expressions have DIFFERENT capabilities.**

### Template Expressions (in tool inputs, agent prompts)
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

**Example - CORRECT (workaround using agent):**
```json
{
  "id": "compute_row",
  "type": "agent",
  "inputs": {
    "model": "qwen/qwen3-235b-a22b-2507",
    "prompt": "Calculate and return only the number: ${index} + 2"
  },
  "outputs": {"mode": "text"}
}
```
Then use `${compute_row}` in subsequent nodes.

### Condition Expressions (in `if` node conditions)
Conditions support a limited expression language — NOT full Python.

**✅ Supported:**
- Arithmetic: `${count} + 1 > 5`, `${price} * ${quantity} > 1000`
- Comparisons: `${status} == 'active'`, `${count} >= 10`
- Boolean logic: `${flag} && ${other}`, `!${disabled}`, `${a} || ${b}`
- Containment: `'value' in ${array}`, `'key' in ${object}`
- Functions: `len(${items}) > 0`, `sum(${values}) < 100`, `any(${flags})`

**Available Functions (whitelist — no others work):** `len`, `any`, `all`, `min`, `max`, `sum`, `str`, `int`, `float`

**❌ NOT Supported in Conditions:**
- Python ternary: `x if cond else y` — use an `if` node with conditional edges instead
- Method calls: `.lower()`, `.strip()`, `.iloc[]`, `.append()`, `.split()`
- Library/module calls: `pd.notna()`, `json.loads()`, `re.match()`, `datetime.now()`
- Comprehensions: `[x for x in items]` — use a `for_each` node instead
- Lambda functions
- Slicing with complex syntax: `[:, 1]` — only simple `[start:end]` on arrays

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

**Pattern 1: Tool → Agent (Process then Analyze)**
```json
{
  "nodes": [
    {"id": "fetch", "type": "tool", "tool": "gmail_get_message", "inputs": {"message_id": "${email_trigger.data.id}"}},
    {"id": "analyze", "type": "agent", "inputs": {"model": "qwen/qwen3-235b-a22b-2507", "prompt": "Analyze: ${fetch}"}, "outputs": {"mode": "json", ...}}
  ],
  "edges": [
    {"source": "fetch", "target": "analyze"}
  ]
}
```

**Pattern 2: Agent → If (Classify then Route)**
```json
{
  "nodes": [
    {"id": "classify", "type": "agent", ...},
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
    {"id": "complete", "type": "agent", "inputs": {"model": "qwen/qwen3-235b-a22b-2507", "prompt": "Summarize: processed all items"}, "outputs": {"mode": "text"}}
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
| `mcp` | External MCP tool | `id`, `server`, `tool` | `${node_id}`, `${node_id.field}` |
| `if` | Conditional branch | `id`, `condition` | N/A (routing only) |
| `for_each` | Loop over list | `id`, `items` | Loop state (internal) |
| `hitl` | Human-In-The-Loop | `id`, `title` | `${node_id.input_field_id}` |
| `image_gen` | Generate images | `id`, `inputs.model`, `inputs.prompt` | `${node_id}` (image URL) |
| `browser` | Browser automation | `id`, `task` | `${node_id}`, `${node_id.field}` |
| `agent` | Autonomous task execution | `id`, `inputs.model`, `inputs.prompt` | `${node_id}`, `${node_id.field}` |

---
