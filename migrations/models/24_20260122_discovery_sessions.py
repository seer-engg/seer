from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "workflow_discovery_chat_sessions" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "thread_id" VARCHAR(255) NOT NULL UNIQUE,
            "title" VARCHAR(255),
            "workflow_creation_mode" VARCHAR(20) NOT NULL DEFAULT 'ASK_FIRST',
            "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
            "created_workflow_id" INT REFERENCES "workflows" ("id") ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_wf_discovery_chat_sessions_user_id" ON "workflow_discovery_chat_sessions" ("user_id");
        CREATE INDEX IF NOT EXISTS "idx_wf_discovery_chat_sessions_thread_id" ON "workflow_discovery_chat_sessions" ("thread_id");"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "workflow_discovery_chat_sessions";"""
