# Nexus Chat API Integration Guide

## Overview

The Nexus Chat API provides an intelligent conversational interface for creating and managing workflows. This guide is for frontend developers integrating the Nexus chat functionality into their applications.

**Base Path:** `/api/nexus`
**Authentication:** Bearer token (JWT)
**Content-Type:** `application/json`

## Quick Start

1. **Authenticate:** Obtain a JWT token for your user
2. **Create/Select Workflow:** Get a workflow ID to work with
3. **Start Chat Session:** Send your first message to begin interaction
4. **Handle Proposals:** Present workflow proposals to users for approval
5. **Accept/Reject:** Apply changes or request modifications

## Authentication

All endpoints require authentication via Bearer token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

## Core Concepts

### Chat Sessions
- Each conversation is tracked as a **session** (stored in database)
- Each session has a **thread_id** (used by LangGraph for state management)
- Sessions persist across page reloads and contain full message history

### Workflow Proposals
- When the agent designs a workflow, it creates a **proposal**
- Proposals contain the full workflow specification
- Users can **accept** (apply changes) or **reject** (request modifications)
- Proposals are versioned and stored for audit trail

### Agent Thinking
- The agent returns **thinking steps** showing its reasoning process
- Use these for transparency: "Searching for email tools...", "Designing workflow..."
- Display thinking steps as loading indicators or debug info

## API Endpoints

### 1. Send Chat Message

**Endpoint:** `POST /api/nexus/{workflow_id}/chat`

Start or continue a conversation with the Nexus agent.

**Request:**
```json
{
  "message": "Create a workflow that sends me a Slack notification when I receive an urgent email",
  "workflow_state": {
    "nodes": [],
    "edges": []
  },
  "session_id": 123,
  "thread_id": "workflow-abc-xyz-thread-1",
  "model": "gpt-4o-mini"
}
```

**Parameters:**
- `message` (required): User's message text
- `workflow_state` (required): Current workflow specification
- `session_id` (optional): Session ID from previous message (for multi-turn)
- `thread_id` (optional): Thread ID from previous message (for multi-turn)
- `model` (optional): LLM model to use (default: gpt-4o-mini)

**Response:**
```json
{
  "response": "I'll help you create that workflow. First, let me search for email and Slack tools...",
  "proposal": {
    "id": 456,
    "spec": {
      "nodes": [
        {
          "id": "trigger_1",
          "type": "trigger",
          "trigger_key": "gmail.new_email",
          "config": {
            "filters": {
              "importance": "high"
            }
          }
        },
        {
          "id": "action_1",
          "type": "tool",
          "tool": "slack_send_message",
          "config": {
            "channel": "#alerts",
            "message": "New urgent email: {{trigger.subject}}"
          }
        }
      ],
      "edges": [
        {
          "from": "trigger_1",
          "to": "action_1"
        }
      ]
    },
    "summary": "Workflow that monitors Gmail for urgent emails and sends Slack notifications",
    "created_at": "2026-01-20T10:00:00Z"
  },
  "session_id": 123,
  "thread_id": "workflow-abc-xyz-thread-1",
  "thinking": [
    "Analyzing user request for email monitoring and Slack notifications",
    "Searching for Gmail trigger capabilities",
    "Searching for Slack messaging tools",
    "Designing workflow structure with trigger and action"
  ]
}
```

**Response Fields:**
- `response`: Agent's text response to display to user
- `proposal`: Workflow proposal (if agent created one), null otherwise
- `session_id`: Session ID (use for subsequent messages)
- `thread_id`: Thread ID (use for subsequent messages)
- `thinking`: Array of agent's reasoning steps
- `proposal_error`: Error message if proposal validation failed

**Error Responses:**
- `400`: Invalid request (missing required fields)
- `401`: Unauthorized (invalid/missing token)
- `404`: Workflow not found
- `422`: Validation error (malformed data)

### 2. Create Chat Session

**Endpoint:** `POST /api/nexus/{workflow_id}/chat/sessions`

Explicitly create a chat session before sending messages (optional - sessions are auto-created).

**Request:**
```json
{}
```

**Response:**
```json
{
  "id": 123,
  "thread_id": "workflow-abc-xyz-thread-1",
  "workflow_id": "workflow-abc",
  "created_at": "2026-01-20T10:00:00Z",
  "updated_at": "2026-01-20T10:00:00Z"
}
```

### 3. List Chat Sessions

**Endpoint:** `GET /api/nexus/{workflow_id}/chat/sessions`

Get all chat sessions for a workflow.

**Response:**
```json
[
  {
    "id": 123,
    "thread_id": "workflow-abc-xyz-thread-1",
    "workflow_id": "workflow-abc",
    "created_at": "2026-01-20T10:00:00Z",
    "updated_at": "2026-01-20T10:05:00Z"
  },
  {
    "id": 124,
    "thread_id": "workflow-abc-xyz-thread-2",
    "workflow_id": "workflow-abc",
    "created_at": "2026-01-20T11:00:00Z",
    "updated_at": "2026-01-20T11:00:00Z"
  }
]
```

### 4. Get Session with Messages

**Endpoint:** `GET /api/nexus/{workflow_id}/chat/sessions/{session_id}`

Retrieve a session with its complete message history.

**Response:**
```json
{
  "id": 123,
  "thread_id": "workflow-abc-xyz-thread-1",
  "workflow_id": "workflow-abc",
  "created_at": "2026-01-20T10:00:00Z",
  "updated_at": "2026-01-20T10:05:00Z",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "Create a workflow for email alerts",
      "created_at": "2026-01-20T10:00:00Z"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "I'll help you create that workflow...",
      "created_at": "2026-01-20T10:00:15Z",
      "proposal_id": 456
    }
  ]
}
```

### 5. Resume After Interrupt

**Endpoint:** `POST /api/nexus/{workflow_id}/chat/resume`

Resume a conversation after an interrupt (e.g., agent needs user input).

**Note:** This is an advanced feature. Most workflows won't require interrupts.

**Request:**
```json
{
  "thread_id": "workflow-abc-xyz-thread-1",
  "user_input": "Yes, proceed with that configuration"
}
```

### 6. Get Workflow Proposal

**Endpoint:** `GET /api/nexus/{workflow_id}/proposals/{proposal_id}`

Retrieve a specific proposal by ID.

**Response:**
```json
{
  "id": 456,
  "workflow_id": "workflow-abc",
  "spec": {
    "nodes": [...],
    "edges": [...]
  },
  "summary": "Workflow that monitors Gmail for urgent emails",
  "status": "pending",
  "created_at": "2026-01-20T10:00:00Z",
  "updated_at": "2026-01-20T10:00:00Z"
}
```

### 7. Accept Workflow Proposal

**Endpoint:** `POST /api/nexus/{workflow_id}/proposals/{proposal_id}/accept`

Accept a proposal and apply its changes to the workflow.

**Request:**
```json
{}
```

**Response:**
```json
{
  "success": true,
  "message": "Workflow proposal accepted and applied",
  "workflow_graph": {
    "id": "workflow-abc",
    "name": "Email Alert Workflow",
    "spec": {
      "nodes": [...],
      "edges": [...]
    },
    "updated_at": "2026-01-20T10:05:00Z"
  }
}
```

### 8. Reject Workflow Proposal

**Endpoint:** `POST /api/nexus/{workflow_id}/proposals/{proposal_id}/reject`

Reject a proposal (optionally provide feedback for revision).

**Request:**
```json
{
  "feedback": "I want the trigger to run every hour instead of on email"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Workflow proposal rejected",
  "feedback_recorded": true
}
```

## Common Workflows

### Basic Chat Interaction

```javascript
// 1. Send initial message
const response1 = await fetch(`/api/nexus/${workflowId}/chat`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: "Create a workflow that emails me daily",
    workflow_state: currentWorkflowSpec
  })
});

const data1 = await response1.json();

// 2. Display response
console.log(data1.response);

// 3. Save session info for next message
const sessionId = data1.session_id;
const threadId = data1.thread_id;

// 4. Send follow-up message
const response2 = await fetch(`/api/nexus/${workflowId}/chat`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: "Make it send at 9am",
    workflow_state: currentWorkflowSpec,
    session_id: sessionId,
    thread_id: threadId
  })
});
```

### Handling Proposals

```javascript
const response = await fetch(`/api/nexus/${workflowId}/chat`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: "Create a Slack notification workflow",
    workflow_state: currentWorkflowSpec
  })
});

const data = await response.json();

// Check if proposal was created
if (data.proposal) {
  const proposal = data.proposal;

  // Show proposal to user
  console.log("Proposal:", proposal.summary);
  console.log("Spec:", proposal.spec);

  // User decides to accept
  const acceptResponse = await fetch(
    `/api/nexus/${workflowId}/proposals/${proposal.id}/accept`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }
  );

  const acceptData = await acceptResponse.json();

  if (acceptData.success) {
    // Update UI with new workflow spec
    updateWorkflow(acceptData.workflow_graph.spec);
  }
}
```

### Displaying Thinking Steps

```javascript
const data = await response.json();

// Show thinking as loading indicators
if (data.thinking && data.thinking.length > 0) {
  data.thinking.forEach(step => {
    console.log(`🤔 ${step}`);
  });
}

// Or display as progress:
// [1/4] Analyzing user request...
// [2/4] Searching for tools...
// [3/4] Designing workflow...
// [4/4] Validating specification...
```

## Error Handling

### Handle Network Errors

```javascript
try {
  const response = await fetch(`/api/nexus/${workflowId}/chat`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      message: userMessage,
      workflow_state: currentWorkflowSpec
    })
  });

  if (!response.ok) {
    const error = await response.json();

    if (response.status === 401) {
      // Redirect to login
      redirectToLogin();
    } else if (response.status === 404) {
      // Workflow not found
      showError("Workflow not found");
    } else if (response.status === 422) {
      // Validation error
      showError(`Validation error: ${error.detail}`);
    } else {
      // Generic error
      showError("Failed to send message");
    }
    return;
  }

  const data = await response.json();
  handleChatResponse(data);

} catch (error) {
  console.error("Network error:", error);
  showError("Could not connect to server");
}
```

### Handle Proposal Errors

```javascript
const data = await response.json();

// Check for proposal validation error
if (data.proposal_error) {
  showWarning(`The workflow couldn't be created: ${data.proposal_error}`);
  showMessage("You can ask me to try a different approach.");
}

// Agent response should also explain the issue
console.log(data.response);
```

## UI/UX Best Practices

### 1. Chat Interface

- Display messages in a chat-like interface (user on right, agent on left)
- Show thinking steps as loading indicators or collapsed details
- Highlight proposals prominently (different background color, call-to-action buttons)
- Keep session_id and thread_id in memory for the session

### 2. Proposal Display

- Show proposal summary at the top
- Visualize the workflow spec (node graph or step-by-step list)
- Provide clear "Accept" and "Reject" buttons
- Allow users to view spec details before accepting
- Consider a diff view if modifying existing workflow

### 3. Loading States

- Show typing indicator while waiting for response
- Display thinking steps as they relate to progress
- Disable input while processing
- Show timeout warning if response takes >30s

### 4. Error Recovery

- Allow users to retry failed messages
- Preserve message history even after errors
- Suggest alternative phrasings if agent doesn't understand
- Provide "start over" option to clear session

### 5. Session Management

- Auto-save session_id and thread_id to localStorage
- Allow users to view and switch between past sessions
- Show timestamps for messages
- Export conversation history

## Testing

### Test Chat Endpoint

```bash
curl -X POST http://localhost:8000/api/nexus/my-workflow-id/chat \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a workflow that sends me a daily email",
    "workflow_state": {"nodes": [], "edges": []}
  }'
```

### Test Proposal Acceptance

```bash
curl -X POST http://localhost:8000/api/nexus/my-workflow-id/proposals/123/accept \
  -H "Authorization: Bearer test-token"
```

## Advanced Features

### Streaming Responses

Currently, responses are returned as complete JSON. Streaming support may be added in future versions.

### Custom Models

You can specify different LLM models in the chat request:

```json
{
  "message": "Create workflow",
  "workflow_state": {},
  "model": "gpt-4o"
}
```

Available models: `gpt-4o`, `gpt-4o-mini`, `claude-3-5-sonnet-20241022`

### Context Window Management

The agent automatically manages conversation history. Very long conversations may be summarized to fit within model context limits.

## Troubleshooting

### "Unauthorized" Error
- Check that JWT token is valid and not expired
- Ensure Authorization header is properly formatted
- Verify user has access to the workflow

### "Workflow not found" Error
- Confirm workflow_id exists and belongs to the user
- Check that workflow hasn't been deleted

### Proposal Not Generated
- Agent may need more information - send follow-up messages
- Check thinking steps to see what agent is doing
- Try more specific requests: "Create a workflow with trigger X and action Y"

### Session Lost After Reload
- Ensure you're saving session_id and thread_id
- Check that you're passing them in subsequent requests
- Use GET /chat/sessions/{session_id} to retrieve history

## Rate Limits

- Chat endpoint: 60 requests per minute per user
- Other endpoints: 120 requests per minute per user

## Support

For issues or questions:
- Check logs for detailed error messages
- Review thinking steps for agent reasoning
- Contact backend team for API issues
