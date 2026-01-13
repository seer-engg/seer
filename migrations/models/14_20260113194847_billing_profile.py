from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "billing_profiles" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "type" VARCHAR(10) NOT NULL DEFAULT 'individual',
    "stripe_customer_id" VARCHAR(255) UNIQUE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "owner_user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
COMMENT ON COLUMN "billing_profiles"."type" IS 'INDIVIDUAL: individual\nTEAM: team';
COMMENT ON TABLE "billing_profiles" IS 'Billing profile for a paying entity (individual user or team).';
        CREATE TABLE IF NOT EXISTS "billing_subscriptions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "stripe_subscription_id" VARCHAR(255) UNIQUE,
    "tier" VARCHAR(8) NOT NULL DEFAULT 'free',
    "status" VARCHAR(10) NOT NULL DEFAULT 'active',
    "current_period_start" TIMESTAMPTZ,
    "current_period_end" TIMESTAMPTZ,
    "cancel_at_period_end" BOOL NOT NULL DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "billing_profile_id" INT NOT NULL UNIQUE REFERENCES "billing_profiles" ("id") ON DELETE CASCADE
);
COMMENT ON COLUMN "billing_subscriptions"."tier" IS 'FREE: free\nPRO: pro\nPRO_PLUS: pro_plus\nULTRA: ultra';
COMMENT ON COLUMN "billing_subscriptions"."status" IS 'ACTIVE: active\nCANCELED: canceled\nPAST_DUE: past_due\nTRIALING: trialing\nINCOMPLETE: incomplete';
COMMENT ON TABLE "billing_subscriptions" IS 'Subscription record tied to a billing profile.';
        DROP TABLE IF EXISTS "user_subscriptions";"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "billing_subscriptions";
        DROP TABLE IF EXISTS "billing_profiles";"""


MODELS_STATE = (
    "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"
)
