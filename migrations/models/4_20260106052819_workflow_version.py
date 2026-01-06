from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "workflows" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT,
    "tags" JSONB,
    "meta" JSONB,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "published_version_id" INT,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "workflows" IS 'New workflow entity that owns drafts, versions, and published state.';
        CREATE TABLE IF NOT EXISTS "workflow_versions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "status" VARCHAR(20) NOT NULL DEFAULT 'DRAFT',
    "spec" JSONB NOT NULL,
    "created_from_draft_revision" INT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "manifest" JSONB,
    "spec_hash" VARCHAR(64),
    "version_number" INT,
    "created_by_id" INT REFERENCES "users" ("id") ON DELETE CASCADE,
    "workflow_id" INT NOT NULL REFERENCES "workflows" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_workflow_ve_workflo_016195" UNIQUE ("workflow_id", "version_number")
);
COMMENT ON COLUMN "workflow_versions"."status" IS 'DRAFT: DRAFT
RELEASED: RELEASED
ARCHIVED: ARCHIVED';
COMMENT ON TABLE "workflow_versions" IS 'Immutable runnable workflow version.';
        CREATE TABLE IF NOT EXISTS "workflow_drafts" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "spec" JSONB NOT NULL,
    "revision" INT NOT NULL DEFAULT 1,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "validation_errors" JSONB,
    "validation_warnings" JSONB,
    "updated_by_id" INT REFERENCES "users" ("id") ON DELETE CASCADE,
    "workflow_id" INT NOT NULL UNIQUE REFERENCES "workflows" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "workflow_drafts" IS 'Mutable draft state for a workflow.';
        ALTER TABLE "workflows" ADD CONSTRAINT "fk_workflows_publish_f6c8e3c5" FOREIGN KEY ("published_version_id") REFERENCES "workflow_versions" ("id") ON DELETE CASCADE;
        ALTER TABLE "workflow_runs" ADD "thread_id" VARCHAR(255);
        ALTER TABLE "workflow_runs" DROP COLUMN IF EXISTS "workflow_version";
        ALTER TABLE "workflow_runs" ADD "workflow_version_id" INT;
        ALTER TABLE "workflow_runs" ADD CONSTRAINT "fk_workflow_workflow_045a5119" FOREIGN KEY ("workflow_version_id") REFERENCES "workflow_versions" ("id") ON DELETE CASCADE;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "workflow_runs" DROP CONSTRAINT IF EXISTS "fk_workflow_workflow_045a5119";
        ALTER TABLE "workflow_runs" DROP COLUMN IF EXISTS "workflow_version_id";
        ALTER TABLE "workflow_runs" ADD "workflow_version" INT;
        ALTER TABLE "workflow_runs" DROP COLUMN "thread_id";
        ALTER TABLE "workflows" DROP CONSTRAINT IF EXISTS "fk_workflows_publish_f6c8e3c5";
        DROP TABLE IF EXISTS "workflow_drafts";
        DROP TABLE IF EXISTS "workflow_versions";
        DROP TABLE IF EXISTS "workflows";"""


MODELS_STATE = (
    "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"
)
