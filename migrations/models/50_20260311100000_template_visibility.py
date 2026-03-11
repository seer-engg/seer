from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE workflow_templates ADD COLUMN IF NOT EXISTS organization_id INT NULL REFERENCES organizations(id) ON DELETE SET NULL;
        ALTER TABLE workflow_templates ADD COLUMN IF NOT EXISTS visibility VARCHAR(10) NOT NULL DEFAULT 'private';
        CREATE INDEX IF NOT EXISTS idx_templates_org_visibility ON workflow_templates(organization_id, visibility);
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS idx_templates_org_visibility;
        ALTER TABLE workflow_templates DROP COLUMN IF EXISTS visibility;
        ALTER TABLE workflow_templates DROP COLUMN IF EXISTS organization_id;
    """
