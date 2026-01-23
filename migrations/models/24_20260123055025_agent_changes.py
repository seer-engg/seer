from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "user_settings" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "max_agent_steps" INT,
    "preferences" JSONB NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL UNIQUE REFERENCES "users" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "user_settings" IS 'Database model for per-user settings.';
        CREATE TABLE IF NOT EXISTS "workflow_discovery_chat_sessions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "thread_id" VARCHAR(255) NOT NULL UNIQUE,
    "title" VARCHAR(255),
    "workflow_creation_mode" VARCHAR(20) NOT NULL DEFAULT 'ASK_FIRST',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_workflow_id" INT REFERENCES "workflows" ("id") ON DELETE CASCADE,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_workflow_di_thread__8ee6f2" ON "workflow_discovery_chat_sessions" ("thread_id");
CREATE INDEX IF NOT EXISTS "idx_workflow_di_user_id_110102" ON "workflow_discovery_chat_sessions" ("user_id");
COMMENT ON COLUMN "workflow_discovery_chat_sessions"."thread_id" IS 'LangGraph thread ID for discovery session';
COMMENT ON COLUMN "workflow_discovery_chat_sessions"."title" IS 'Optional title for discovery session';
COMMENT ON COLUMN "workflow_discovery_chat_sessions"."workflow_creation_mode" IS 'How workflow should be created';
COMMENT ON COLUMN "workflow_discovery_chat_sessions"."created_workflow_id" IS 'Workflow created from this discovery session';
COMMENT ON TABLE "workflow_discovery_chat_sessions" IS 'Discovery chat session before workflow creation.';
        ALTER TABLE "users" ADD "default_workflow_creation_mode" VARCHAR(20) NOT NULL DEFAULT 'ASK_FIRST';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "users" DROP COLUMN "default_workflow_creation_mode";
        DROP TABLE IF EXISTS "user_settings";
        DROP TABLE IF EXISTS "workflow_discovery_chat_sessions";"""


MODELS_STATE = (
    "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"
    "osM9NTfblR0RkXu+slzZe40H//3/0hcewA=="
)
