# Bug Fix: Clarification Questions Disappearing (Dashboard → Canvas)

**Issue ID:** Clarification questions disappearing when initiated from dashboard
**Date Fixed:** 2026-02-01
**Severity:** High - Core feature broken

## Problem Description

When a user entered a chat message in the dashboard that triggered clarification questions from the agent:
- The clarification questions would briefly appear (flash) in the canvas chat panel
- Then immediately disappear, making it impossible to answer them
- Direct canvas chat worked fine - only dashboard → canvas flow was affected

## Root Cause

A **race condition** between local message state and backend message fetching caused clarification question data to be lost:

### The Race Condition Flow

1. **Dashboard sends message** (`useDiscoveryChat.ts`)
   - Creates new workflow
   - Clears local messages
   - Navigates to canvas with `initialMessage` in state

2. **Canvas mounts and auto-sends** (`useInitialChatMessage.ts`)
   - Picks up `initialMessage` from navigation state
   - Calls `handleSend()` to send the message

3. **API responds with clarification question** (`useChatActions.ts`)
   - Response includes `interrupt_required` and `interrupt_data` fields
   - Creates assistant message with clarification question
   - **Adds to local messages** ✅
   - Sets `currentSessionId` from response

4. **Query activates** (`useChatSessionData.ts` - THE BUG)
   - When `currentSessionId` changes from `null` → `number`, `useChatMessages` query activates
   - Fetches messages from backend
   - Effect runs: `setMessages(sessionMessages)` → **OVERWRITES local messages** ❌
   - Local clarification question data is lost!

### Why Direct Canvas Chat Worked

When chatting directly in the canvas:
- Session already exists from the start
- No sudden query activation when `currentSessionId` is set
- No race condition between local and backend messages

## Solution

### Fix 1: Preserve `interrupt_data` in Backend Messages
**File:** `src/hooks/useChatMessages.ts`

Added missing fields to `SessionMessageResponse` type and ensured they're mapped when fetching from backend:

```typescript
type SessionMessageResponse = {
  // ... existing fields
  interrupt_required?: boolean;
  interrupt_data?: {
    type?: string;
    clarification_question?: ClarificationQuestion;
    [key: string]: unknown;
  };
};

// In mapping:
return response.messages.map((msg) => ({
  // ... other fields
  interruptRequired: msg.interrupt_required,
  interruptData: msg.interrupt_data,
}));
```

### Fix 2: Protect Local Clarification Questions
**File:** `src/hooks/useChatSessionData.ts`

Implemented smart overwrite protection that prevents backend messages from clobbering local state during critical windows:

```typescript
// PROTECTION RULES (order matters):
// 1. Skip if no backend messages fetched yet
// 2. Skip if currently sending a message (backend hasn't persisted yet)
// 3. Skip if we have local clarification questions (precious data!)
// 4. Only update when truly switching sessions OR initial empty load

const hasLocalInterruptData = messages.some(
  (msg) => msg.interruptRequired && msg.interruptData
);

if (currentlySending || hasLocalInterruptData) {
  return; // Protect local state!
}
```

## Files Changed

1. **`src/hooks/useChatMessages.ts`**
   - Added `interrupt_required` and `interrupt_data` to type
   - Preserved these fields when mapping backend messages
   - Added comprehensive documentation

2. **`src/hooks/useChatSessionData.ts`**
   - Implemented smart overwrite protection
   - Added session switching detection
   - Added extensive comments explaining the race condition

3. **`src/hooks/useChatActions.ts`**
   - Added comment about importance of preserving interrupt fields

4. **`src/hooks/useInitialChatMessage.ts`**
   - Added documentation explaining dashboard → canvas flow
   - Cleaned up debug logs

## Testing

**Before Fix:**
- Dashboard message → clarification questions flash and disappear ❌

**After Fix:**
- Dashboard message → clarification questions persist ✅
- Canvas direct chat → still works ✅
- Session switching → loads correctly ✅

## Prevention

To prevent similar issues in the future:

1. **Never blindly overwrite chat messages** - always check for precious local data first
2. **Preserve ALL fields from backend** - especially interrupt/clarification data
3. **Consider race conditions** when syncing local and backend state
4. **Test both dashboard and canvas flows** - they have different initialization paths

## Related Code Patterns

This fix applies to any scenario where:
- Local state has fresher data than backend (before persistence completes)
- Query refetch could overwrite local changes
- Timing-sensitive data needs protection during state transitions

Similar protection may be needed for:
- Proposal preview data
- Temporary UI state during async operations
- Any "interrupt" or "pending response" patterns
