from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "projects" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "project_name" VARCHAR(255) NOT NULL UNIQUE,
    "description" TEXT,
    "metadata" JSONB,
    "is_active" BOOL NOT NULL DEFAULT True,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE "projects" IS 'Database model for projects.';
CREATE TABLE IF NOT EXISTS "users" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "user_id" VARCHAR(255) NOT NULL UNIQUE,
    "email" VARCHAR(320),
    "first_name" VARCHAR(255),
    "last_name" VARCHAR(255),
    "claims" JSONB,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_users_user_id_a795d9" ON "users" ("user_id");
COMMENT ON TABLE "users" IS 'Database model for authenticated users.';
CREATE TABLE IF NOT EXISTS "oauth_connections" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "provider" VARCHAR(50) NOT NULL,
    "provider_account_id" VARCHAR(255) NOT NULL,
    "access_token_enc" TEXT NOT NULL,
    "refresh_token_enc" TEXT,
    "expires_at" TIMESTAMPTZ,
    "scopes" TEXT,
    "token_type" VARCHAR(50) NOT NULL DEFAULT 'Bearer',
    "provider_metadata" JSONB,
    "status" VARCHAR(20) NOT NULL DEFAULT 'active',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_oauth_conne_user_id_edb955" UNIQUE ("user_id", "provider", "provider_account_id")
);
COMMENT ON TABLE "oauth_connections" IS 'Database model for storing OAuth connections/tokens.';
CREATE TABLE IF NOT EXISTS "trigger_events" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "trigger_key" VARCHAR(255) NOT NULL,
    "provider_connection_id" INT,
    "provider_event_id" VARCHAR(255),
    "occurred_at" TIMESTAMPTZ,
    "received_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "event" JSONB NOT NULL,
    "raw_payload" JSONB,
    "status" VARCHAR(20) NOT NULL DEFAULT 'received',
    "error" JSONB,
    CONSTRAINT "uid_trigger_eve_trigger_b917c2" UNIQUE ("trigger_key", "provider_connection_id", "provider_event_id")
);
CREATE INDEX IF NOT EXISTS "idx_trigger_eve_status_9e17ee" ON "trigger_events" ("status", "received_at");
CREATE INDEX IF NOT EXISTS "idx_trigger_eve_trigger_c2e9c3" ON "trigger_events" ("trigger_key", "provider_connection_id");
COMMENT ON COLUMN "trigger_events"."status" IS 'RECEIVED: received\nROUTED: routed\nPROCESSED: processed\nFAILED: failed';
COMMENT ON TABLE "trigger_events" IS 'Normalized incoming trigger event.';
CREATE TABLE IF NOT EXISTS "workflow_records" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT,
    "spec" JSONB NOT NULL,
    "version" INT NOT NULL DEFAULT 1,
    "tags" JSONB,
    "meta" JSONB,
    "last_compile_ok" BOOL NOT NULL DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "workflow_records" IS 'Normalized workflow entity backed by WorkflowSpec JSON.';
CREATE TABLE IF NOT EXISTS "trigger_subscriptions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "trigger_key" VARCHAR(255) NOT NULL,
    "provider_connection_id" INT,
    "enabled" BOOL NOT NULL DEFAULT True,
    "filters" JSONB,
    "bindings" JSONB,
    "provider_config" JSONB,
    "secret_token" VARCHAR(255),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "workflow_id" INT NOT NULL REFERENCES "workflow_records" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_trigger_sub_user_id_5c5218" ON "trigger_subscriptions" ("user_id", "workflow_id");
CREATE INDEX IF NOT EXISTS "idx_trigger_sub_trigger_d2ffa4" ON "trigger_subscriptions" ("trigger_key", "provider_connection_id", "enabled");
COMMENT ON TABLE "trigger_subscriptions" IS 'Trigger configuration attached to a workflow.';
CREATE TABLE IF NOT EXISTS "workflow_chat_sessions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "thread_id" VARCHAR(255) NOT NULL UNIQUE,
    "title" VARCHAR(255),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "workflow_id" INT NOT NULL REFERENCES "workflow_records" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_workflow_ch_thread__42ac7b" ON "workflow_chat_sessions" ("thread_id");
COMMENT ON TABLE "workflow_chat_sessions" IS 'Chat session for workflow assistant.';
CREATE TABLE IF NOT EXISTS "workflow_proposals" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "summary" VARCHAR(512) NOT NULL,
    "spec" JSONB NOT NULL,
    "status" VARCHAR(20) NOT NULL DEFAULT 'pending',
    "preview_graph" JSONB,
    "applied_graph" JSONB,
    "metadata" JSONB,
    "decided_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "session_id" INT REFERENCES "workflow_chat_sessions" ("id") ON DELETE CASCADE,
    "workflow_id" INT NOT NULL REFERENCES "workflow_records" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "workflow_proposals" IS 'Reviewable workflow edit proposal.';
CREATE TABLE IF NOT EXISTS "workflow_chat_messages" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "role" VARCHAR(20) NOT NULL,
    "content" TEXT NOT NULL,
    "thinking" TEXT,
    "suggested_edits" JSONB,
    "metadata" JSONB,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "session_id" INT NOT NULL REFERENCES "workflow_chat_sessions" ("id") ON DELETE CASCADE,
    "proposal_id" INT UNIQUE REFERENCES "workflow_proposals" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "workflow_chat_messages" IS 'Individual message in a chat session.';
CREATE TABLE IF NOT EXISTS "workflow_runs" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "workflow_version" INT,
    "spec" JSONB NOT NULL,
    "inputs" JSONB,
    "config" JSONB,
    "source" VARCHAR(20) NOT NULL DEFAULT 'manual',
    "status" VARCHAR(20) NOT NULL DEFAULT 'queued',
    "output" JSONB,
    "error" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "started_at" TIMESTAMPTZ,
    "finished_at" TIMESTAMPTZ,
    "metrics" JSONB,
    "subscription_id" INT REFERENCES "trigger_subscriptions" ("id") ON DELETE CASCADE,
    "trigger_event_id" INT REFERENCES "trigger_events" ("id") ON DELETE CASCADE,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "workflow_id" INT REFERENCES "workflow_records" ("id") ON DELETE CASCADE
);
COMMENT ON COLUMN "workflow_runs"."source" IS 'MANUAL: manual\nTRIGGER: trigger';
COMMENT ON COLUMN "workflow_runs"."status" IS 'QUEUED: queued\nRUNNING: running\nSUCCEEDED: succeeded\nFAILED: failed\nCANCELLED: cancelled';
COMMENT ON TABLE "workflow_runs" IS 'Persisted workflow run metadata (no telemetry).';
CREATE TABLE IF NOT EXISTS "aerich" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "version" VARCHAR(255) NOT NULL,
    "app" VARCHAR(100) NOT NULL,
    "content" JSONB NOT NULL
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """


MODELS_STATE = (
    "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"
    "1+Xh7/AWfWVh0="
)
