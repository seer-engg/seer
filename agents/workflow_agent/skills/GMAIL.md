# Gmail Integration Guide

Gmail integration provides comprehensive email automation capabilities including drafts, sending, reading, and management.

## Available Tools (15 total)

### Draft Operations (5 tools)

#### **gmail_create_draft**
Create a Gmail draft with full formatting support.

**Parameters:**
- `to` (required): array of recipient emails
- `subject` (required): email subject
- `body_text` (required): plain text body
- `body_html` (optional): HTML body for rich formatting
- `cc`, `bcc` (optional): CC and BCC recipients
- `from_email` (optional): sender email (must be allowed alias)
- `reply_to` (optional): reply-to address
- `in_reply_to`, `references` (optional): threading headers
- `attachments` (optional): array of {filename, mime_type, data_base64}

**Returns:** `{id, message: {id, threadId}}`

**Example usage:**
```json
{
  "tool": "gmail_create_draft",
  "in": {
    "to": ["${user.email}"],
    "subject": "Welcome to our platform!",
    "body_text": "Hi ${user.name},\n\nWelcome aboard!\n\nBest,\nThe Team"
  }
}
```

#### **gmail_send_draft**
Send an existing draft by ID.

**Parameters:**
- `draft_id` (required): ID of draft to send
- `update_raw_message` (optional): update content before sending

**Use when:** You want to create a draft for review before sending

#### **gmail_list_drafts**
List all drafts with filtering.

**Parameters:**
- `max_results` (default: 10, max: 100): number of results
- `q` (optional): Gmail query to filter drafts
- `page_token` (optional): for pagination
- `include_spam_trash` (default: false): include spam/trash

#### **gmail_get_draft** & **gmail_delete_draft**
Retrieve or delete specific drafts by ID.

---

### Sending Email (1 tool)

#### **gmail_send_email**
Send email immediately (without draft step).

**Parameters:** Same as `gmail_create_draft`

**Returns:** `{id, threadId, labelIds}`

**Use when:** You want to send email directly without creating a draft first

**Example usage:**
```json
{
  "tool": "gmail_send_email",
  "in": {
    "to": ["customer@example.com"],
    "subject": "Order Confirmation #${order.id}",
    "body_text": "Thank you for your order!\n\nOrder details:\n${order.summary}",
    "cc": ["sales@example.com"]
  }
}
```

---

### Reading Email (5 tools)

#### **gmail_read_emails**
Search and read emails with filters.

**Parameters:**
- `max_results` (default: 10, max: 100)
- `label_ids` (default: ["INBOX"]): filter by labels
- `q` (optional): Gmail search query
- `include_body` (default: false): fetch full email body

**Gmail Query Syntax:**
- `is:unread` - Unread messages
- `from:user@example.com` - From specific sender
- `subject:"invoice"` - Subject contains word
- `after:2024/01/01` - Date filtering
- `has:attachment` - Has attachments
- `in:inbox` - In specific folder

**Example usage:**
```json
{
  "tool": "gmail_read_emails",
  "in": {
    "q": "from:support@example.com is:unread",
    "max_results": 20,
    "include_body": true
  }
}
```

#### **gmail_get_message**
Get specific email by message ID with full details.

#### **gmail_list_threads** & **gmail_get_thread**
Work with email threads (conversations).

#### **gmail_get_attachment**
Download attachment from a message.

---

### Management Operations (4 tools)

#### **gmail_modify_message_labels**
Add or remove labels from messages (mark as read, archive, etc.).

#### **gmail_trash_message** & **gmail_delete_message**
Move to trash or permanently delete messages.

#### **gmail_list_labels**, **gmail_create_label**, **gmail_delete_label**
Manage custom labels/folders.

---

## Common Workflow Patterns

### Pattern 1: New Supabase Signup → Welcome Draft
Automatically create welcome email draft when user signs up.

```json
{
  "triggers": [{
    "key": "webhook.supabase.db_changes",
    "config": {
      "table": "users",
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
      "id": "create_draft",
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

### Pattern 2: Incoming Email → Process with LLM → Reply Draft
Process support emails and draft intelligent replies.

```json
{
  "triggers": [{
    "key": "poll.gmail.email_received",
    "config": {
      "q": "from:support@example.com is:unread"
    }
  }],
  "nodes": [
    {
      "id": "classify_email",
      "type": "llm",
      "model": "gpt-5-mini",
      "prompt": "Classify this email: ${trigger.data.message.body}",
      "output": {
        "mode": "json",
        "schema": {
          "json_schema": {
            "type": "object",
            "properties": {
              "category": {"type": "string"},
              "priority": {"type": "string"},
              "suggested_response": {"type": "string"}
            }
          }
        }
      },
      "out": "classification"
    },
    {
      "id": "create_reply_draft",
      "type": "tool",
      "tool": "gmail_create_draft",
      "in": {
        "to": ["${trigger.data.message.from}"],
        "subject": "Re: ${trigger.data.message.subject}",
        "body_text": "${classification.suggested_response}",
        "in_reply_to": "${trigger.data.message.id}",
        "thread_id": "${trigger.data.message.threadId}"
      }
    }
  ]
}
```

### Pattern 3: Scheduled Daily Report via Email
Generate and send daily report on schedule.

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
      "id": "generate_report",
      "type": "llm",
      "model": "gpt-5-mini",
      "prompt": "Generate daily summary for ${trigger.data.scheduled_time}",
      "out": "report"
    },
    {
      "id": "send_report",
      "type": "tool",
      "tool": "gmail_send_email",
      "in": {
        "to": ["team@example.com"],
        "subject": "Daily Report - ${trigger.data.scheduled_time}",
        "body_text": "${report}"
      }
    }
  ]
}
```

### Pattern 4: Form Submission → Notification Email
Send alert when form is submitted.

```json
{
  "triggers": [{
    "key": "form.hosted.submission",
    "config": {"form_id": "contact_form"}
  }],
  "nodes": [
    {
      "id": "send_alert",
      "type": "tool",
      "tool": "gmail_send_email",
      "in": {
        "to": ["admin@example.com"],
        "subject": "New Form: ${trigger.data.submission.name}",
        "body_text": "Name: ${trigger.data.submission.name}\nEmail: ${trigger.data.submission.email}\nMessage: ${trigger.data.submission.message}"
      }
    }
  ]
}
```

---

## Best Practices

### Email Formatting
- Always provide `body_text` (plain text) for compatibility
- Use `body_html` for rich formatting, tables, images
- Test emails with both HTML and plain text clients

### Threading & Replies
- Use `in_reply_to` and `references` headers to maintain proper threading
- Include `thread_id` when replying to keep conversations grouped

### Draft vs Send
- **Use drafts** when human review is needed before sending
- **Use send** for automated notifications where review isn't required
- Create draft → review → call `gmail_send_draft` for approval workflows

### Query Optimization
- Use specific Gmail queries to reduce API calls
- Combine filters: `from:support@example.com is:unread after:2024/01/01`
- Set `max_results` appropriately (higher = more API usage)

### Error Handling
- Check that email addresses are valid before sending
- Handle attachment size limits (25MB for Gmail)
- Verify OAuth scopes are sufficient for operation

### Trigger Data Access
- **Supabase trigger**: `${trigger.data.record}` contains new row
- **Gmail poll trigger**: `${trigger.data.message}` contains email data
- **Form trigger**: `${trigger.data.submission}` contains form fields
- **Schedule trigger**: `${trigger.data.scheduled_time}` contains timestamp

---

## Required OAuth Scopes

- **gmail.readonly**: Read emails, list drafts
- **gmail.compose**: Create and send drafts
- **gmail.send**: Send emails directly
- **gmail.modify**: Modify labels, trash/delete messages

Ensure integration connection has appropriate scopes for the tools you're using.
