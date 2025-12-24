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
CREATE TABLE IF NOT EXISTS "workflows" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT,
    "graph_data" JSONB NOT NULL,
    "schema_version" VARCHAR(50) NOT NULL DEFAULT '1.0',
    "is_active" BOOL NOT NULL DEFAULT True,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "workflows" IS 'Main workflow entity.';
CREATE TABLE IF NOT EXISTS "workflow_blocks" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "block_id" VARCHAR(255) NOT NULL,
    "block_type" VARCHAR(100) NOT NULL,
    "block_config" JSONB NOT NULL,
    "python_code" TEXT,
    "position_x" DOUBLE PRECISION NOT NULL,
    "position_y" DOUBLE PRECISION NOT NULL,
    "oauth_scope" VARCHAR(255),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "workflow_id" INT NOT NULL REFERENCES "workflows" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_workflow_bl_workflo_650105" UNIQUE ("workflow_id", "block_id")
);
COMMENT ON TABLE "workflow_blocks" IS 'Individual blocks (nodes) in workflow.';
CREATE TABLE IF NOT EXISTS "workflow_chat_sessions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "thread_id" VARCHAR(255) NOT NULL UNIQUE,
    "title" VARCHAR(255),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "workflow_id" INT NOT NULL REFERENCES "workflows" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_workflow_ch_thread__42ac7b" ON "workflow_chat_sessions" ("thread_id");
COMMENT ON TABLE "workflow_chat_sessions" IS 'Chat session for workflow assistant.';
CREATE TABLE IF NOT EXISTS "workflow_chat_messages" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "role" VARCHAR(20) NOT NULL,
    "content" TEXT NOT NULL,
    "thinking" TEXT,
    "suggested_edits" JSONB,
    "metadata" JSONB,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "session_id" INT NOT NULL REFERENCES "workflow_chat_sessions" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "workflow_chat_messages" IS 'Individual message in a chat session.';
CREATE TABLE IF NOT EXISTS "workflow_edges" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "source_handle" VARCHAR(100),
    "target_handle" VARCHAR(100),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "source_block_id" INT NOT NULL REFERENCES "workflow_blocks" ("id") ON DELETE CASCADE,
    "target_block_id" INT NOT NULL REFERENCES "workflow_blocks" ("id") ON DELETE CASCADE,
    "workflow_id" INT NOT NULL REFERENCES "workflows" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "workflow_edges" IS 'Connections between blocks.';
CREATE TABLE IF NOT EXISTS "workflow_executions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "status" VARCHAR(50) NOT NULL,
    "input_data" JSONB,
    "output_data" JSONB,
    "error_message" TEXT,
    "started_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completed_at" TIMESTAMPTZ,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "workflow_id" INT NOT NULL REFERENCES "workflows" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "workflow_executions" IS 'Workflow execution history.';
CREATE TABLE IF NOT EXISTS "block_executions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "status" VARCHAR(50) NOT NULL,
    "input_data" JSONB,
    "output_data" JSONB,
    "error_message" TEXT,
    "execution_time_ms" INT,
    "started_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completed_at" TIMESTAMPTZ,
    "block_id" INT NOT NULL REFERENCES "workflow_blocks" ("id") ON DELETE CASCADE,
    "execution_id" INT NOT NULL REFERENCES "workflow_executions" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "block_executions" IS 'Per-block execution logs.';
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
    "gPD4P7gNyJU="
)
