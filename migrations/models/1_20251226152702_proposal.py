from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "workflow_proposals" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "summary" VARCHAR(512) NOT NULL,
    "patch_ops" JSONB NOT NULL,
    "status" VARCHAR(20) NOT NULL DEFAULT 'pending',
    "preview_graph" JSONB,
    "applied_graph" JSONB,
    "metadata" JSONB,
    "decided_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "session_id" INT REFERENCES "workflow_chat_sessions" ("id") ON DELETE CASCADE,
    "workflow_id" INT NOT NULL REFERENCES "workflows" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "workflow_proposals" IS 'Reviewable workflow edit proposal.';
        ALTER TABLE "workflow_chat_messages" ADD "proposal_id" INT UNIQUE;
        ALTER TABLE "workflow_chat_messages" ADD CONSTRAINT "fk_workflow_workflow_9fa660a4" FOREIGN KEY ("proposal_id") REFERENCES "workflow_proposals" ("id") ON DELETE CASCADE;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "workflow_chat_messages" DROP CONSTRAINT IF EXISTS "fk_workflow_workflow_9fa660a4";
        ALTER TABLE "workflow_chat_messages" DROP COLUMN "proposal_id";
        DROP TABLE IF EXISTS "workflow_proposals";"""


MODELS_STATE = (
    "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"
    "hMgxNYFucovF6/8Bz67YOQ=="
)
