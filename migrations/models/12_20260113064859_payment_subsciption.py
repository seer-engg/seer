from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "user_subscriptions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "stripe_customer_id" VARCHAR(255) UNIQUE,
    "stripe_subscription_id" VARCHAR(255) UNIQUE,
    "tier" VARCHAR(8) NOT NULL DEFAULT 'free',
    "status" VARCHAR(10) NOT NULL DEFAULT 'active',
    "current_period_start" TIMESTAMPTZ,
    "current_period_end" TIMESTAMPTZ,
    "cancel_at_period_end" BOOL NOT NULL DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL UNIQUE REFERENCES "users" ("id") ON DELETE CASCADE
);
COMMENT ON COLUMN "user_subscriptions"."tier" IS 'FREE: free\nPRO: pro\nPRO_PLUS: pro_plus\nULTRA: ultra';
COMMENT ON COLUMN "user_subscriptions"."status" IS 'ACTIVE: active\nCANCELED: canceled\nPAST_DUE: past_due\nTRIALING: trialing\nINCOMPLETE: incomplete';
COMMENT ON TABLE "user_subscriptions" IS 'Tracks user subscription state from Stripe.';

ALTER TABLE trigger_subscriptions
ADD CONSTRAINT IF NOT EXISTS trigger_subscriptions_workflow_id_fkey
FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE;
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "user_subscriptions";"""


MODELS_STATE = (
    "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"
)
