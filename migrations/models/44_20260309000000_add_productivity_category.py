from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True

CATEGORY_COMMENT_NEW = (
    "MARKETING: marketing\\n"
    "CUSTOMER_SUPPORT: customer_support\\n"
    "SALES: sales\\n"
    "PRODUCTIVITY: productivity"
)

CATEGORY_COMMENT_OLD = (
    "MARKETING: marketing\\n"
    "CUSTOMER_SUPPORT: customer_support\\n"
    "SALES: sales"
)


async def upgrade(db: BaseDBAsyncClient) -> str:
    return (
        'COMMENT ON COLUMN "workflow_templates"."category"'
        f" IS '{CATEGORY_COMMENT_NEW}';"
    )


async def downgrade(db: BaseDBAsyncClient) -> str:
    return (
        'COMMENT ON COLUMN "workflow_templates"."category"'
        f" IS '{CATEGORY_COMMENT_OLD}';"
    )
