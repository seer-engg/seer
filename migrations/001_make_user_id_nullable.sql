-- Migration: Make user_id nullable in workflows and workflow_executions tables
-- This supports both self-hosted (NULL user_id) and cloud (set user_id) modes

-- Make user_id nullable in workflows table
ALTER TABLE workflows ALTER COLUMN user_id DROP NOT NULL;

-- Make user_id nullable in workflow_executions table
ALTER TABLE workflow_executions ALTER COLUMN user_id DROP NOT NULL;

-- Note: Indexes remain unchanged as they support NULL values

