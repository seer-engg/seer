# Supabase Integration Guide

Supabase integration provides database operations, auth management, storage, edge functions, and real-time triggers.

## Available Tools (15 total)

### Database Operations (6 tools)

#### **supabase_table_query**
Query (read) rows from a Supabase table.

**Parameters:**
- `integration_resource_id` (required): Supabase project resource ID
- `table` (required): Table or view name
- `select` (default: "*"): Columns to select (PostgREST syntax)
- `filters` (optional): Object mapping columns to values (eq filter)
- `limit` (default: 100, max: 1000): Max rows to return
- `order_by` (optional): Ordering, e.g., "created_at.desc" or "id.asc"

**Returns:** Array of row objects

**Example usage:**
```json
{
  "tool": "supabase_table_query",
  "in": {
    "integration_resource_id": 123,
    "table": "users",
    "select": "id,email,name,created_at",
    "filters": {
      "status": "active"
    },
    "order_by": "created_at.desc",
    "limit": 50
  }
}
```

#### **supabase_table_insert**
Insert new row(s) into a Supabase table.

**Parameters:**
- `integration_resource_id` (required): Supabase project resource ID
- `table` (required): Table name
- `values` (required): Object or array of objects to insert
- `select` (optional): Columns to return after insert

**Returns:** Inserted row(s) data

**Example usage:**
```json
{
  "tool": "supabase_table_insert",
  "in": {
    "integration_resource_id": 123,
    "table": "audit_log",
    "values": {
      "user_id": "${user.id}",
      "action": "signup",
      "timestamp": "${trigger.data.scheduled_time}",
      "metadata": {"source": "workflow"}
    }
  }
}
```

#### **supabase_table_update**
Update existing rows in a Supabase table.

**Parameters:**
- `integration_resource_id` (required)
- `table` (required)
- `values` (required): Fields to update
- `filters` (required): Which rows to update (eq filters)
- `select` (optional): Columns to return

#### **supabase_table_upsert**
Insert or update rows (insert if not exists, update if exists).

**Parameters:** Same as insert, plus:
- `on_conflict` (optional): Column(s) to check for conflicts

#### **supabase_table_delete**
Delete rows from a Supabase table.

**Parameters:**
- `integration_resource_id` (required)
- `table` (required)
- `filters` (required): Which rows to delete

**⚠️ Warning:** Deletion is permanent. Use filters carefully.

#### **supabase_rpc_call**
Call a Supabase RPC (Remote Procedure Call / stored function).

**Parameters:**
- `integration_resource_id` (required)
- `function_name` (required): Name of the PostgreSQL function
- `args` (optional): Object with function arguments

---

### Auth Admin (3 tools)

#### **supabase_auth_admin_list_users**
List users in Supabase Auth.

#### **supabase_auth_admin_create_user**
Create a new auth user programmatically.

#### **supabase_auth_admin_delete_user**
Delete an auth user by ID.

---

### Storage (5 tools)

#### **supabase_storage_list_buckets**
List all storage buckets in the project.

#### **supabase_storage_create_bucket**
Create a new storage bucket.

#### **supabase_storage_upload_object**
Upload file to a storage bucket.

#### **supabase_storage_download_object**
Download file from storage.

#### **supabase_storage_create_signed_object_url**
Generate signed URL for private object access.

---

### Edge Functions (1 tool)

#### **supabase_function_invoke**
Invoke a Supabase Edge Function.

**Parameters:**
- `integration_resource_id` (required)
- `function_name` (required): Name of the edge function
- `payload` (optional): JSON payload to send

---

## Triggers

### **webhook.supabase.db_changes**
Real-time database changes trigger.

**Configuration:**
- `integration_resource_id` (required): Supabase project ID
- `table` (required): Table name to watch
- `schema` (default: "public"): Database schema
- `events` (required): Array of event types: ["INSERT"], ["UPDATE"], ["DELETE"], or ["INSERT", "UPDATE", "DELETE"]

**Trigger Data:**
```json
{
  "trigger.data.record": {}, // New row data (INSERT, UPDATE)
  "trigger.data.old_record": {}, // Old row data (UPDATE, DELETE only)
  "trigger.data.type": "INSERT", // Event type
  "trigger.data.table": "users", // Table name
  "trigger.data.schema": "public" // Schema name
}
```

**Example configuration:**
```json
{
  "key": "webhook.supabase.db_changes",
  "config": {
    "integration_resource_id": 123,
    "table": "orders",
    "schema": "public",
    "events": ["INSERT", "UPDATE"]
  }
}
```

---

## Common Workflow Patterns

### Pattern 1: New Signup → Welcome Email Draft
Detect new user signup and create Gmail welcome draft.

```json
{
  "version": "1",
  "triggers": [{
    "key": "webhook.supabase.db_changes",
    "config": {
      "integration_resource_id": 123,
      "table": "users",
      "schema": "public",
      "events": ["INSERT"]
    }
  }],
  "nodes": [
    {
      "id": "extract_user",
      "type": "task",
      "kind": "set",
      "value": "${trigger.data.record}",
      "out": "user"
    },
    {
      "id": "create_welcome_draft",
      "type": "tool",
      "tool": "gmail_create_draft",
      "in": {
        "to": ["${user.email}"],
        "subject": "Welcome ${user.name}!",
        "body_text": "Hi ${user.name},\n\nWelcome to our platform!"
      }
    }
  ]
}
```

### Pattern 2: Update Tracking → Log to Audit Table
Track all updates to sensitive table by logging to audit table.

```json
{
  "triggers": [{
    "key": "webhook.supabase.db_changes",
    "config": {
      "integration_resource_id": 123,
      "table": "sensitive_data",
      "events": ["UPDATE"]
    }
  }],
  "nodes": [
    {
      "id": "log_change",
      "type": "tool",
      "tool": "supabase_table_insert",
      "in": {
        "integration_resource_id": 123,
        "table": "audit_log",
        "values": {
          "record_id": "${trigger.data.record.id}",
          "old_value": "${trigger.data.old_record}",
          "new_value": "${trigger.data.record}",
          "changed_at": "${trigger.data.timestamp}"
        }
      }
    }
  ]
}
```

### Pattern 3: Daily Report → Query Data → Send Email
Generate daily report from database queries.

```json
{
  "triggers": [{
    "key": "schedule.cron",
    "config": {
      "cron_expression": "0 9 * * *",
      "timezone": "America/New_York"
    }
  }],
  "nodes": [
    {
      "id": "query_metrics",
      "type": "tool",
      "tool": "supabase_table_query",
      "in": {
        "integration_resource_id": 123,
        "table": "daily_metrics",
        "select": "*",
        "filters": {
          "date": "${trigger.data.scheduled_time}"
        },
        "limit": 100
      },
      "out": "metrics"
    },
    {
      "id": "generate_summary",
      "type": "llm",
      "model": "gpt-5-mini",
      "prompt": "Summarize these metrics:\n${metrics}",
      "out": "summary"
    },
    {
      "id": "send_report",
      "type": "tool",
      "tool": "gmail_send_email",
      "in": {
        "to": ["team@example.com"],
        "subject": "Daily Metrics Report",
        "body_text": "${summary}"
      }
    }
  ]
}
```

### Pattern 4: Form Submission → Store in Supabase
Capture form submissions and store in database.

```json
{
  "triggers": [{
    "key": "form.hosted.submission",
    "config": {"form_id": "contact_form"}
  }],
  "nodes": [
    {
      "id": "store_submission",
      "type": "tool",
      "tool": "supabase_table_insert",
      "in": {
        "integration_resource_id": 123,
        "table": "form_submissions",
        "values": {
          "name": "${trigger.data.submission.name}",
          "email": "${trigger.data.submission.email}",
          "message": "${trigger.data.submission.message}",
          "submitted_at": "${trigger.data.submission.timestamp}"
        }
      }
    }
  ]
}
```

### Pattern 5: Conditional Update → Check Status → Update Row
Update row based on certain conditions.

```json
{
  "nodes": [
    {
      "id": "query_order",
      "type": "tool",
      "tool": "supabase_table_query",
      "in": {
        "integration_resource_id": 123,
        "table": "orders",
        "filters": {"id": "${inputs.order_id}"},
        "limit": 1
      },
      "out": "order"
    },
    {
      "id": "check_status",
      "type": "condition",
      "condition": "${order[0].status} == 'pending'",
      "out": "should_update"
    },
    {
      "id": "update_status",
      "type": "tool",
      "tool": "supabase_table_update",
      "in": {
        "integration_resource_id": 123,
        "table": "orders",
        "filters": {"id": "${inputs.order_id}"},
        "values": {"status": "processing", "updated_at": "${now()}"}
      },
      "condition": "${should_update}"
    }
  ]
}
```

---

## Best Practices

### Query Optimization
- Use `select` to fetch only needed columns (reduces bandwidth)
- Set appropriate `limit` (default 100, max 1000)
- Use `filters` for simple equality checks
- For complex queries, consider `supabase_rpc_call` with stored functions

### Trigger Data Access
- **INSERT events**: `${trigger.data.record}` contains the new row
- **UPDATE events**: `${trigger.data.record}` (new data), `${trigger.data.old_record}` (previous data)
- **DELETE events**: `${trigger.data.old_record}` contains deleted row
- **All events**: `${trigger.data.type}` contains event type ("INSERT", "UPDATE", "DELETE")

### Security Considerations
- Tool execution uses service role key (bypasses RLS)
- Be careful with delete operations - always use specific filters
- Validate user input before inserting into database
- Use RPC functions for complex business logic with proper validation

### Error Handling
- Check that `integration_resource_id` is valid and connected
- Verify table names match exactly (case-sensitive)
- Ensure filters match existing columns
- Handle cases where queries return empty arrays

### PostgREST Query Syntax
Supabase uses PostgREST for REST API. Advanced usage:

**Select syntax:**
- `"id,name,email"` - specific columns
- `"*"` - all columns
- `"id,profile(*)"` - with related table
- `"id,profile(name,avatar)"` - nested select

**Filters (via filters object):**
- `{"status": "active"}` becomes `?status=eq.active`
- For complex filters, use direct PostgREST query parameters

**Order by:**
- `"created_at.desc"` - descending
- `"id.asc"` - ascending
- `"created_at.desc,id.asc"` - multiple columns

---

## Required Configuration

### Integration Resource
- Create Supabase project connection in Seer platform
- Note the `integration_resource_id` for use in workflows
- Ensure service role key is stored securely

### Database Webhooks (for triggers)
- Supabase project must have webhooks enabled
- Configure webhook URL in Supabase dashboard to point to Seer
- Webhooks fire in real-time when database changes occur

### Row Level Security (RLS)
- Service role key bypasses RLS policies
- Ensure workflows only access data they should
- Consider adding workflow-specific validation logic
