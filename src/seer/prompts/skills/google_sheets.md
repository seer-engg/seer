# Google Sheets Integration Guide

Google Sheets integration provides spreadsheet automation including reading, writing, appending data, and managing spreadsheet structure.

## Available Tools (9 total)

### Read Operations (3 tools)

#### **google_sheets_read**
Read data from a single range in a Google Sheet.

**Parameters:**
- `spreadsheet_id` (required): Google Sheets spreadsheet ID
- `range` (required): A1 notation range (e.g., "Sheet1!A1:D10", "Sheet1")
- `major_dimension` (optional): "ROWS" or "COLUMNS" (default: "ROWS")
- `value_render_option` (optional): "FORMATTED_VALUE", "UNFORMATTED_VALUE", or "FORMULA"
- `date_time_render_option` (optional): "SERIAL_NUMBER" or "FORMATTED_STRING"

**Returns:**
```json
{
  "range": "Sheet1!A1:D100",
  "majorDimension": "ROWS",
  "values": [
    ["Header1", "Header2", "Header3"],
    ["Row1Val1", "Row1Val2", "Row1Val3"],
    ["Row2Val1", "Row2Val2", "Row2Val3"]
  ]
}
```

**Example usage:**
```json
{
  "id": "read_data",
  "type": "tool",
  "tool": "google_sheets_read",
  "inputs": {
    "spreadsheet_id": "1abc123xyz",
    "range": "Sheet1!A1:D100"
  }
}
```

#### **google_sheets_batch_read**
Read data from multiple ranges in a single request.

**Parameters:**
- `spreadsheet_id` (required): Spreadsheet ID
- `ranges` (required): Array of A1 notation ranges

**Example:**
```json
{
  "inputs": {
    "spreadsheet_id": "1abc123xyz",
    "ranges": ["Sheet1!A1:B10", "Sheet2!A1:C5"]
  }
}
```

#### **google_sheets_get_spreadsheet**
Get spreadsheet metadata (sheet names, properties).

**Parameters:**
- `spreadsheet_id` (required): Spreadsheet ID
- `include_grid_data` (optional): Include cell data (default: false)
- `ranges` (optional): Specific ranges to include

---

### Write Operations (6 tools)

#### **google_sheets_write**
Write data to a specific range, overwriting existing data.

**Parameters:**
- `spreadsheet_id` (required): Spreadsheet ID
- `range` (required): A1 notation range (e.g., "Sheet1!A1")
- `values` (required): 2D array of values
- `value_input_option` (optional): "RAW" or "USER_ENTERED" (default)

**Example:**
```json
{
  "id": "write_results",
  "type": "tool",
  "tool": "google_sheets_write",
  "inputs": {
    "spreadsheet_id": "1abc123xyz",
    "range": "Results!A1",
    "values": [
      ["Name", "Score", "Status"],
      ["${user.name}", "${analysis.score}", "${analysis.status}"]
    ]
  }
}
```

#### **google_sheets_append**
Append rows after existing data (most common for logging/tracking).

**Parameters:**
- `spreadsheet_id` (required): Spreadsheet ID
- `range` (required): A1 notation (data appended after last row in range)
- `values` (required): 2D array of rows to append
- `value_input_option` (optional): "RAW" or "USER_ENTERED"
- `insert_data_option` (optional): "OVERWRITE" or "INSERT_ROWS" (default)

**Example - Log each processed item:**
```json
{
  "id": "log_result",
  "type": "tool",
  "tool": "google_sheets_append",
  "inputs": {
    "spreadsheet_id": "1abc123xyz",
    "range": "Logs!A1",
    "values": [
      ["${item.timestamp}", "${item.email}", "${item.status}"]
    ]
  }
}
```

#### **google_sheets_clear**
Clear data from a range (keeps formatting).

**Parameters:**
- `spreadsheet_id` (required): Spreadsheet ID
- `range` (required): A1 notation range to clear

#### **google_sheets_batch_write**
Write to multiple ranges in one request.

**Parameters:**
- `spreadsheet_id` (required): Spreadsheet ID
- `data` (required): Array of {range, values} objects
- `value_input_option` (optional): "RAW" or "USER_ENTERED"

#### **google_sheets_create_spreadsheet**
Create a new spreadsheet.

**Parameters:**
- `title` (required): Spreadsheet name
- `sheets` (optional): Initial sheet configuration array

#### **google_sheets_batch_update_spreadsheet**
Advanced: Update formatting, add sheets, merge cells.

**Parameters:**
- `spreadsheet_id` (required): Spreadsheet ID
- `requests` (required): Array of Request objects (Google Sheets API format)
- `include_spreadsheet_in_response` (optional): Return updated spreadsheet

---

## Working with Spreadsheet Row Data (CRITICAL)

### Understanding the Values Array

`google_sheets_read` returns data as a **2D array** (array of arrays):

```json
{
  "values": [
    ["Name", "Email", "Status"],      // Row 0 (header)
    ["John", "john@example.com", "Active"],   // Row 1
    ["Jane", "jane@example.com", "Pending"]   // Row 2
  ]
}
```

### The Array Indexing Problem

When you iterate over `${read_data.values}` with `for_each`, each `item` is an **array** (a row):
```
item = ["John", "john@example.com", "Active"]
```

**⚠️ PROBLEM:** Direct array indexing like `${item[0]}` may NOT work reliably!

This is because:
1. The `for_each` loop registers `item` with a permissive object schema at compile time
2. The compiler doesn't know `item` is actually an array
3. Array indexing validation may fail

### Recommended Solution: Use LLM to Extract Values

Instead of trying to access array indices directly, use an LLM node to parse the row:

```json
{
  "id": "parse_row",
  "type": "agent",
  "inputs": {
    "model": "openai/gpt-oss-120b",
    "prompt": "Parse this spreadsheet row: ${item}\n\nExpected columns:\n- Index 0: Name\n- Index 1: Email\n- Index 2: Status\n\nReturn as JSON with name, email, status fields."
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

Now you can access `${parse_row.name}`, `${parse_row.email}`, `${parse_row.status}` reliably!

---

## Common Workflow Patterns

### Pattern 1: Read Spreadsheet and Process Each Row

**Use Case:** Process a list of contacts from a spreadsheet

```json
{
  "version": "2",
  "nodes": [
    {
      "id": "read_contacts",
      "type": "tool",
      "tool": "google_sheets_read",
      "inputs": {
        "spreadsheet_id": "1abc123xyz",
        "range": "Contacts!A2:C100"
      }
    },
    {
      "id": "loop_rows",
      "type": "for_each",
      "items": "${read_contacts.values}",
      "item_var": "row"
    },
    {
      "id": "parse_contact",
      "type": "agent",
      "inputs": {
        "model": "openai/gpt-oss-120b",
        "prompt": "Parse spreadsheet row: ${row}. Columns: Name (0), Email (1), Company (2). Return JSON."
      },
      "outputs": {
        "mode": "json",
        "schema": {
          "json_schema": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "email": {"type": "string"},
              "company": {"type": "string"}
            },
            "required": ["name", "email"]
          }
        }
      }
    },
    {
      "id": "send_email",
      "type": "tool",
      "tool": "gmail_send_email",
      "inputs": {
        "to": ["${parse_contact.email}"],
        "subject": "Hello ${parse_contact.name}!",
        "body_text": "Thanks for joining from ${parse_contact.company}."
      }
    },
    {
      "id": "complete",
      "type": "agent",
      "inputs": {
        "model": "openai/gpt-oss-120b",
        "prompt": "Summarize: All contacts have been emailed."
      },
      "outputs": {"mode": "text"}
    }
  ],
  "edges": [
    {"source": "read_contacts", "target": "loop_rows"},
    {"source": "loop_rows", "target": "parse_contact", "type": "loop_body"},
    {"source": "parse_contact", "target": "send_email"},
    {"source": "send_email", "target": "loop_rows"},
    {"source": "loop_rows", "target": "complete", "type": "loop_exit"}
  ]
}
```

**Key Points:**
- Skip header row by starting range at A2 instead of A1
- Use LLM to extract named fields from array rows
- The `loop_exit` points to an existing node (`complete`)

### Pattern 2: Database Event → Log to Spreadsheet

**Use Case:** Log new database records to a tracking spreadsheet

```json
{
  "version": "2",
  "triggers": [
    {
      "id": "new_order",
      "key": "webhook.supabase.db_changes",
      "mode": "webhook",
      "event_schema": {},
      "meta": {"requires_connection": true},
      "provider_config": {"table": "orders", "events": ["INSERT"]}
    }
  ],
  "nodes": [
    {
      "id": "log_to_sheet",
      "type": "tool",
      "tool": "google_sheets_append",
      "inputs": {
        "spreadsheet_id": "1xyz789abc",
        "range": "Orders!A1",
        "values": [
          [
            "${new_order.data.record.id}",
            "${new_order.data.record.customer_email}",
            "${new_order.data.record.total}",
            "${new_order.data.record.created_at}"
          ]
        ]
      }
    }
  ],
  "edges": [
    {"source": "new_order", "target": "log_to_sheet", "type": "trigger"}
  ]
}
```

### Pattern 3: Scheduled Report → Spreadsheet

**Use Case:** Daily summary written to spreadsheet

```json
{
  "version": "2",
  "triggers": [
    {
      "id": "daily_schedule",
      "key": "schedule.cron",
      "mode": "polling",
      "event_schema": {},
      "provider_config": {"cron_expression": "0 9 * * *", "timezone": "America/New_York"}
    }
  ],
  "nodes": [
    {
      "id": "generate_summary",
      "type": "agent",
      "inputs": {
        "model": "openai/gpt-oss-120b",
        "prompt": "Generate daily metrics summary for ${daily_schedule.data.scheduled_time}. Include: date, revenue, orders, customers."
      },
      "outputs": {
        "mode": "json",
        "schema": {
          "json_schema": {
            "type": "object",
            "properties": {
              "date": {"type": "string"},
              "revenue": {"type": "number"},
              "orders": {"type": "integer"},
              "customers": {"type": "integer"}
            }
          }
        }
      }
    },
    {
      "id": "write_report",
      "type": "tool",
      "tool": "google_sheets_append",
      "inputs": {
        "spreadsheet_id": "1reports123",
        "range": "DailyMetrics!A1",
        "values": [
          [
            "${generate_summary.date}",
            "${generate_summary.revenue}",
            "${generate_summary.orders}",
            "${generate_summary.customers}"
          ]
        ]
      }
    }
  ],
  "edges": [
    {"source": "daily_schedule", "target": "generate_summary", "type": "trigger"},
    {"source": "generate_summary", "target": "write_report"}
  ]
}
```

### Pattern 4: Form Submission → Add Row

**Use Case:** Form responses logged to spreadsheet

```json
{
  "version": "2",
  "triggers": [
    {
      "id": "contact_form",
      "key": "form.hosted",
      "mode": "webhook",
      "event_schema": {},
      "provider_config": {"form_id": "contact-us"}
    }
  ],
  "nodes": [
    {
      "id": "log_submission",
      "type": "tool",
      "tool": "google_sheets_append",
      "inputs": {
        "spreadsheet_id": "1forms456",
        "range": "Submissions!A1",
        "values": [
          [
            "${contact_form.data.timestamp}",
            "${contact_form.data.name}",
            "${contact_form.data.email}",
            "${contact_form.data.message}"
          ]
        ]
      }
    }
  ],
  "edges": [
    {"source": "contact_form", "target": "log_submission", "type": "trigger"}
  ]
}
```

---

## A1 Notation Reference

| Format | Description | Example |
|--------|-------------|---------|
| `Sheet1` | All data in Sheet1 | Entire sheet |
| `Sheet1!A1:D10` | Range from A1 to D10 | First 10 rows, 4 columns |
| `Sheet1!A:A` | Entire column A | All rows in column |
| `Sheet1!1:1` | Entire row 1 | All columns in first row |
| `Sheet1!A2:D` | From A2 to end | Skip header, all data |
| `'Sheet Name'!A1:B10` | Sheet with spaces | Use quotes around name |

---

## Best Practices

### Spreadsheet ID
- Found in the URL: `https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit`
- Use resource picker in UI or store in workflow config

### Range Selection
- **Skip header rows** when processing data: use `Sheet1!A2:D100` instead of `Sheet1!A1:D100`
- Use open-ended ranges for append: `Sheet1!A1` (appends after last row)

### Error Handling
- Check if `values` array exists and is non-empty before looping
- Use conditional routing for empty data scenarios

### Performance
- Use `batch_read` for multiple ranges instead of multiple `read` calls
- Use `batch_write` when updating multiple ranges
- Use `append` for adding rows (more efficient than read-then-write)

### Data Types
- `value_input_option: "USER_ENTERED"` - Sheets parses values (dates, numbers)
- `value_input_option: "RAW"` - Values stored as-is (text only)

---

## Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| `${item[0]}` | Array indexing may fail in loops | Use LLM to parse row data |
| `${row + 2}` | Arithmetic not allowed in templates | Use LLM to compute values |
| `range: "Sheet1!A${index}"` | Dynamic range with variable | Pre-compute range or use append |
| Empty `values` array | No data but loop still runs | Check `len(${data.values}) > 0` before loop |

---

## Required OAuth Scopes

For read-only operations:
- `https://www.googleapis.com/auth/spreadsheets.readonly`

For read/write operations:
- `https://www.googleapis.com/auth/spreadsheets`
- `https://www.googleapis.com/auth/drive.file`
- `https://www.googleapis.com/auth/drive.metadata.readonly`

Ensure your Google integration connection has appropriate scopes enabled.
