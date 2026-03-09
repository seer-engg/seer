from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "billing_profiles" DROP CONSTRAINT IF EXISTS "billing_profiles_owner_user_id_key";
        DROP INDEX IF EXISTS "uid_billing_pro_owner_u_6077ba";
        ALTER TABLE "billing_profiles" DROP CONSTRAINT IF EXISTS "fk_billing__users_b875e799";
        CREATE TABLE IF NOT EXISTS "organizations" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "name" VARCHAR(255) NOT NULL,
    "slug" VARCHAR(255) NOT NULL UNIQUE,
    "type" VARCHAR(8) NOT NULL DEFAULT 'personal',
    "settings" JSONB NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "owner_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_organizatio_slug_81ed67" ON "organizations" ("slug");
COMMENT ON COLUMN "organizations"."type" IS 'PERSONAL: personal\nTEAM: team';
COMMENT ON TABLE "organizations" IS 'Represents a workspace/team. Can be personal or team.';
        CREATE TABLE IF NOT EXISTS "organization_invitations" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "email" VARCHAR(320) NOT NULL,
    "role" VARCHAR(10) NOT NULL DEFAULT 'user',
    "token" VARCHAR(255) NOT NULL UNIQUE,
    "expires_at" TIMESTAMPTZ NOT NULL,
    "status" VARCHAR(8) NOT NULL DEFAULT 'pending',
    "accepted_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "accepted_by_id" INT REFERENCES "users" ("id") ON DELETE SET NULL,
    "invited_by_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "organization_id" INT NOT NULL REFERENCES "organizations" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_organizatio_email_b33be4" ON "organization_invitations" ("email");
CREATE INDEX IF NOT EXISTS "idx_organizatio_token_66c6d0" ON "organization_invitations" ("token");
CREATE INDEX IF NOT EXISTS "idx_organizatio_organiz_9acf8e" ON "organization_invitations" ("organization_id", "status");
CREATE INDEX IF NOT EXISTS "idx_organizatio_email_bc6a1d" ON "organization_invitations" ("email", "status");
COMMENT ON COLUMN "organization_invitations"."role" IS 'OWNER: owner\nADMIN: admin\nUSER: user\nCONSULTANT: consultant';
COMMENT ON COLUMN "organization_invitations"."status" IS 'PENDING: pending\nACCEPTED: accepted\nEXPIRED: expired\nREVOKED: revoked';
COMMENT ON TABLE "organization_invitations" IS 'Pending invitations (before user accepts/signs up).';
        CREATE TABLE IF NOT EXISTS "organization_memberships" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "role" VARCHAR(10) NOT NULL DEFAULT 'user',
    "status" VARCHAR(9) NOT NULL DEFAULT 'active',
    "invited_at" TIMESTAMPTZ,
    "joined_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "invited_by_id" INT REFERENCES "users" ("id") ON DELETE SET NULL,
    "organization_id" INT NOT NULL REFERENCES "organizations" ("id") ON DELETE CASCADE,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_organizatio_organiz_f44366" UNIQUE ("organization_id", "user_id")
);
CREATE INDEX IF NOT EXISTS "idx_organizatio_user_id_7d8199" ON "organization_memberships" ("user_id", "status");
CREATE INDEX IF NOT EXISTS "idx_organizatio_organiz_2d7d4a" ON "organization_memberships" ("organization_id", "role");
COMMENT ON COLUMN "organization_memberships"."role" IS 'OWNER: owner\nADMIN: admin\nUSER: user\nCONSULTANT: consultant';
COMMENT ON COLUMN "organization_memberships"."status" IS 'PENDING: pending\nACTIVE: active\nSUSPENDED: suspended';
COMMENT ON TABLE "organization_memberships" IS 'User membership in an organization with role.';
        CREATE TABLE IF NOT EXISTS "workflow_approvals" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "requested_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "status" VARCHAR(8) NOT NULL DEFAULT 'pending',
    "reviewed_at" TIMESTAMPTZ,
    "review_notes" TEXT,
    "organization_id" INT NOT NULL REFERENCES "organizations" ("id") ON DELETE CASCADE,
    "requested_by_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "reviewed_by_id" INT REFERENCES "users" ("id") ON DELETE SET NULL,
    "workflow_id" INT NOT NULL REFERENCES "workflows" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_workflow_ap_organiz_9149a8" ON "workflow_approvals" ("organization_id", "status");
CREATE INDEX IF NOT EXISTS "idx_workflow_ap_workflo_c78e81" ON "workflow_approvals" ("workflow_id");
CREATE INDEX IF NOT EXISTS "idx_workflow_ap_request_19aafd" ON "workflow_approvals" ("requested_by_id");
COMMENT ON COLUMN "workflow_approvals"."status" IS 'PENDING: pending\nAPPROVED: approved\nREJECTED: rejected';
COMMENT ON TABLE "workflow_approvals" IS 'Approval requests for consultant-created workflows.';
        CREATE TABLE IF NOT EXISTS "workflow_assignments" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "assigned_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expires_at" TIMESTAMPTZ,
    "assigned_by_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "organization_id" INT NOT NULL REFERENCES "organizations" ("id") ON DELETE CASCADE,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "workflow_id" INT NOT NULL REFERENCES "workflows" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_workflow_as_workflo_45e969" UNIQUE ("workflow_id", "user_id")
);
CREATE INDEX IF NOT EXISTS "idx_workflow_as_user_id_77a3c8" ON "workflow_assignments" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_workflow_as_workflo_47a114" ON "workflow_assignments" ("workflow_id");
CREATE INDEX IF NOT EXISTS "idx_workflow_as_organiz_2ca640" ON "workflow_assignments" ("organization_id");
COMMENT ON TABLE "workflow_assignments" IS 'Workflow assignments for consultants.';
        ALTER TABLE "oauth_connections" ADD "shared_with_organization_id" INT;
        ALTER TABLE "workflows" ADD "approval_status" VARCHAR(20);
        ALTER TABLE "workflows" ADD "organization_id" INT;
        ALTER TABLE "workflows" ADD "visibility" VARCHAR(20) NOT NULL DEFAULT 'team';
        ALTER TABLE "workflow_files" ADD "organization_id" INT;
        ALTER TABLE "billing_profiles" ADD "owner_organization_id" INT;
        ALTER TABLE "billing_profiles" ALTER COLUMN "owner_user_id" DROP NOT NULL;
        ALTER TABLE "llm_usage_records" ADD "organization_id" INT;
        ALTER TABLE "llm_usage_records" ALTER COLUMN "user_id" DROP NOT NULL;
        ALTER TABLE "usage_counters" ADD "organization_id" INT;
        ALTER TABLE "usage_counters" ALTER COLUMN "user_id" DROP NOT NULL;
        ALTER TABLE "knowledge_bases" ADD "organization_id" INT;
        COMMENT ON COLUMN "workflows"."approval_status" IS 'Approval status for consultant-created workflows';
COMMENT ON COLUMN "workflows"."visibility" IS 'Who can see this workflow within the organization';
COMMENT ON COLUMN "workflow_files"."organization_id" IS 'Organization this file belongs to (for team access)';
COMMENT ON COLUMN "knowledge_bases"."organization_id" IS 'Organization this KB belongs to (for team access)';
        ALTER TABLE "oauth_connections" ADD CONSTRAINT "fk_oauth_co_organiza_91ca2018" FOREIGN KEY ("shared_with_organization_id") REFERENCES "organizations" ("id") ON DELETE SET NULL;
        CREATE INDEX IF NOT EXISTS "idx_oauth_conne_user_id_e24902" ON "oauth_connections" ("user_id", "provider");
        CREATE INDEX IF NOT EXISTS "idx_oauth_conne_shared__900783" ON "oauth_connections" ("shared_with_organization_id");
        ALTER TABLE "workflows" ADD CONSTRAINT "fk_workflow_organiza_308aa387" FOREIGN KEY ("organization_id") REFERENCES "organizations" ("id") ON DELETE CASCADE;
        CREATE INDEX IF NOT EXISTS "idx_workflows_user_id_26ed10" ON "workflows" ("user_id", "organization_id");
        CREATE INDEX IF NOT EXISTS "idx_workflows_organiz_a1a39b" ON "workflows" ("organization_id");
        CREATE INDEX IF NOT EXISTS "idx_workflows_visibil_1af75d" ON "workflows" ("visibility");
        ALTER TABLE "workflow_files" ADD CONSTRAINT "fk_workflow_organiza_001088f6" FOREIGN KEY ("organization_id") REFERENCES "organizations" ("id") ON DELETE CASCADE;
        CREATE INDEX IF NOT EXISTS "idx_workflow_fi_organiz_8002a3" ON "workflow_files" ("organization_id");
        ALTER TABLE "billing_profiles" ADD CONSTRAINT "fk_billing__organiza_9aa00351" FOREIGN KEY ("owner_organization_id") REFERENCES "organizations" ("id") ON DELETE CASCADE;
        ALTER TABLE "billing_profiles" ADD CONSTRAINT "fk_billing__users_b875e799" FOREIGN KEY ("owner_user_id") REFERENCES "users" ("id") ON DELETE CASCADE;
        ALTER TABLE "llm_usage_records" ADD CONSTRAINT "fk_llm_usag_organiza_4cf67092" FOREIGN KEY ("organization_id") REFERENCES "organizations" ("id") ON DELETE CASCADE;
        CREATE INDEX IF NOT EXISTS "idx_llm_usage_r_organiz_d40857" ON "llm_usage_records" ("organization_id", "created_at");
        CREATE INDEX IF NOT EXISTS "idx_llm_usage_r_organiz_e8d989" ON "llm_usage_records" ("organization_id");
        ALTER TABLE "usage_counters" ADD CONSTRAINT "fk_usage_co_organiza_7da01c8f" FOREIGN KEY ("organization_id") REFERENCES "organizations" ("id") ON DELETE CASCADE;
        CREATE INDEX IF NOT EXISTS "idx_usage_count_organiz_ed85ae" ON "usage_counters" ("organization_id", "resource_type", "period_start");
        CREATE INDEX IF NOT EXISTS "idx_usage_count_organiz_c524f1" ON "usage_counters" ("organization_id");
        ALTER TABLE "knowledge_bases" ADD CONSTRAINT "fk_knowledg_organiza_21ec8b9c" FOREIGN KEY ("organization_id") REFERENCES "organizations" ("id") ON DELETE CASCADE;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_oauth_conne_shared__900783";
        DROP INDEX IF EXISTS "idx_oauth_conne_user_id_e24902";
        ALTER TABLE "oauth_connections" DROP CONSTRAINT IF EXISTS "fk_oauth_co_organiza_91ca2018";
        DROP INDEX IF EXISTS "idx_llm_usage_r_organiz_e8d989";
        DROP INDEX IF EXISTS "idx_llm_usage_r_organiz_d40857";
        ALTER TABLE "llm_usage_records" DROP CONSTRAINT IF EXISTS "fk_llm_usag_organiza_4cf67092";
        ALTER TABLE "billing_profiles" DROP CONSTRAINT IF EXISTS "fk_billing__users_b875e799";
        ALTER TABLE "billing_profiles" DROP CONSTRAINT IF EXISTS "fk_billing__organiza_9aa00351";
        ALTER TABLE "knowledge_bases" DROP CONSTRAINT IF EXISTS "fk_knowledg_organiza_21ec8b9c";
        DROP INDEX IF EXISTS "idx_workflow_fi_organiz_8002a3";
        ALTER TABLE "workflow_files" DROP CONSTRAINT IF EXISTS "fk_workflow_organiza_001088f6";
        DROP INDEX IF EXISTS "idx_usage_count_organiz_c524f1";
        DROP INDEX IF EXISTS "idx_usage_count_organiz_ed85ae";
        ALTER TABLE "usage_counters" DROP CONSTRAINT IF EXISTS "fk_usage_co_organiza_7da01c8f";
        DROP INDEX IF EXISTS "idx_workflows_visibil_1af75d";
        DROP INDEX IF EXISTS "idx_workflows_organiz_a1a39b";
        DROP INDEX IF EXISTS "idx_workflows_user_id_26ed10";
        ALTER TABLE "workflows" DROP CONSTRAINT IF EXISTS "fk_workflow_organiza_308aa387";
        ALTER TABLE "workflows" DROP COLUMN "approval_status";
        ALTER TABLE "workflows" DROP COLUMN "organization_id";
        ALTER TABLE "workflows" DROP COLUMN "visibility";
        ALTER TABLE "usage_counters" DROP COLUMN "organization_id";
        ALTER TABLE "usage_counters" ALTER COLUMN "user_id" SET NOT NULL;
        ALTER TABLE "workflow_files" DROP COLUMN "organization_id";
        ALTER TABLE "knowledge_bases" DROP COLUMN "organization_id";
        ALTER TABLE "billing_profiles" DROP COLUMN "owner_organization_id";
        ALTER TABLE "billing_profiles" ALTER COLUMN "owner_user_id" SET NOT NULL;
        ALTER TABLE "llm_usage_records" DROP COLUMN "organization_id";
        ALTER TABLE "llm_usage_records" ALTER COLUMN "user_id" SET NOT NULL;
        ALTER TABLE "oauth_connections" DROP COLUMN "shared_with_organization_id";
        DROP TABLE IF EXISTS "workflow_assignments";
        DROP TABLE IF EXISTS "organization_memberships";
        DROP TABLE IF EXISTS "organizations";
        DROP TABLE IF EXISTS "organization_invitations";
        DROP TABLE IF EXISTS "workflow_approvals";
        ALTER TABLE "billing_profiles" ADD CONSTRAINT "fk_billing__users_b875e799" FOREIGN KEY ("owner_user_id") REFERENCES "users" ("id") ON DELETE CASCADE;
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_billing_pro_owner_u_6077ba" ON "billing_profiles" ("owner_user_id");"""


MODELS_STATE = (
    "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"
    "djkreX"
)
