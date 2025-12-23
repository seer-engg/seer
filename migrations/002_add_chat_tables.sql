-- Migration: Add workflow chat session and message tables
-- Created: 2025-01-XX
-- Description: Adds tables for persisting workflow chat assistant conversations

-- Create workflow_chat_sessions table
CREATE TABLE IF NOT EXISTS workflow_chat_sessions (
    id SERIAL PRIMARY KEY,
    workflow_id INTEGER NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    user_id VARCHAR(255),
    thread_id VARCHAR(255) NOT NULL UNIQUE,
    title VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for workflow_chat_sessions
CREATE INDEX IF NOT EXISTS idx_workflow_chat_sessions_workflow_id ON workflow_chat_sessions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_chat_sessions_user_id ON workflow_chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_workflow_chat_sessions_thread_id ON workflow_chat_sessions(thread_id);
CREATE INDEX IF NOT EXISTS idx_workflow_chat_sessions_updated_at ON workflow_chat_sessions(updated_at DESC);

-- Create workflow_chat_messages table
CREATE TABLE IF NOT EXISTS workflow_chat_messages (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES workflow_chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    thinking TEXT,
    suggested_edits JSONB,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for workflow_chat_messages
CREATE INDEX IF NOT EXISTS idx_workflow_chat_messages_session_id ON workflow_chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_chat_messages_created_at ON workflow_chat_messages(created_at);

-- Add comment to tables
COMMENT ON TABLE workflow_chat_sessions IS 'Stores chat sessions for workflow assistant conversations';
COMMENT ON TABLE workflow_chat_messages IS 'Stores individual messages within chat sessions';

