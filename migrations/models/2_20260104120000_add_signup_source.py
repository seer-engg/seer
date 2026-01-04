from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "users" ADD "signup_source" VARCHAR(50);
        COMMENT ON COLUMN "users"."signup_source" IS 'User acquisition channel: HN, Twitter, Reddit, LinkedIn, Organic, Direct';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "users" DROP COLUMN "signup_source";"""


MODELS_STATE = ""
