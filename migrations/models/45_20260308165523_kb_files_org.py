from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Add organization FK to knowledge_bases (nullable for legacy data)
        ALTER TABLE "knowledge_bases" ADD "organization_id" INT;
        ALTER TABLE "knowledge_bases" ADD CONSTRAINT "fk_knowledg_organiza_8b3c5d1f"
            FOREIGN KEY ("organization_id") REFERENCES "organizations" ("id") ON DELETE CASCADE;
        CREATE INDEX IF NOT EXISTS "idx_knowledg_organiz_8b3c5d"
            ON "knowledge_bases" ("organization_id");

        -- Add organization FK to workflow_files (nullable for legacy data)
        ALTER TABLE "workflow_files" ADD "organization_id" INT;
        ALTER TABLE "workflow_files" ADD CONSTRAINT "fk_workflow_organiza_7e4a2c9b"
            FOREIGN KEY ("organization_id") REFERENCES "organizations" ("id") ON DELETE CASCADE;
        CREATE INDEX IF NOT EXISTS "idx_workflow_organiz_7e4a2c"
            ON "workflow_files" ("organization_id");

        -- Backfill: Set organization from user's organization membership
        UPDATE "knowledge_bases" kb
        SET "organization_id" = (
            SELECT om."organization_id"
            FROM "organization_memberships" om
            WHERE om."user_id" = kb."user_id"
            LIMIT 1
        )
        WHERE kb."organization_id" IS NULL;

        UPDATE "workflow_files" wf
        SET "organization_id" = (
            SELECT om."organization_id"
            FROM "organization_memberships" om
            WHERE om."user_id" = wf."user_id"
            LIMIT 1
        )
        WHERE wf."organization_id" IS NULL;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_workflow_organiz_7e4a2c";
        ALTER TABLE "workflow_files" DROP CONSTRAINT IF EXISTS "fk_workflow_organiza_7e4a2c9b";
        ALTER TABLE "workflow_files" DROP COLUMN "organization_id";

        DROP INDEX IF EXISTS "idx_knowledg_organiz_8b3c5d";
        ALTER TABLE "knowledge_bases" DROP CONSTRAINT IF EXISTS "fk_knowledg_organiza_8b3c5d1f";
        ALTER TABLE "knowledge_bases" DROP COLUMN "organization_id";"""
