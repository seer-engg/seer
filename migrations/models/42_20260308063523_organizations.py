from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
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
COMMENT ON TABLE "workflow_assignments" IS 'Workflow assignments for consultants.';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "organization_memberships";
        DROP TABLE IF EXISTS "workflow_approvals";
        DROP TABLE IF EXISTS "organizations";
        DROP TABLE IF EXISTS "workflow_assignments";
        DROP TABLE IF EXISTS "organization_invitations";"""


MODELS_STATE = (
    "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"
    "Zv83j56/8B8quYiA=="
)
