from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE trigger_subscriptions
        ADD CONSTRAINT trigger_subscriptions_workflow_id_fkey
        FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE;
        """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE trigger_subscriptions
        DROP CONSTRAINT IF EXISTS trigger_subscriptions_workflow_id_fkey;
        """


MODELS_STATE = ""
