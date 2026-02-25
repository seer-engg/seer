from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "overage_settings" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "enabled" BOOL NOT NULL DEFAULT False,
    "spending_cap_cents" INT NOT NULL DEFAULT 5000,
    "margin_multiplier" DECIMAL(5,2) NOT NULL DEFAULT 1.30,
    "current_period_overage_cents" INT NOT NULL DEFAULT 0,
    "current_period_start" TIMESTAMPTZ,
    "stripe_metered_subscription_item_id" VARCHAR(255),
    "enabled_at" TIMESTAMPTZ,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "billing_profile_id" INT NOT NULL UNIQUE REFERENCES "billing_profiles" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "overage_settings" IS 'Overage pricing settings for a billing profile.';
        CREATE TABLE IF NOT EXISTS "overage_usage_records" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "base_cost_cents" INT NOT NULL,
    "billed_amount_cents" INT NOT NULL,
    "stripe_usage_record_id" VARCHAR(255),
    "reported_to_stripe_at" TIMESTAMPTZ,
    "status" VARCHAR(8) NOT NULL DEFAULT 'pending',
    "error_message" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "llm_usage_record_id" INT REFERENCES "llm_usage_records" ("id") ON DELETE SET NULL,
    "overage_settings_id" INT NOT NULL REFERENCES "overage_settings" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_overage_usa_overage_bbd911" ON "overage_usage_records" ("overage_settings_id", "status");
CREATE INDEX IF NOT EXISTS "idx_overage_usa_overage_fd2ac0" ON "overage_usage_records" ("overage_settings_id", "created_at");
COMMENT ON COLUMN "overage_usage_records"."status" IS 'PENDING: pending\nREPORTED: reported\nFAILED: failed';
COMMENT ON TABLE "overage_usage_records" IS 'Individual overage usage record for Stripe reporting.';
        COMMENT ON COLUMN "billing_subscriptions"."tier" IS 'FREE: free
PRO: pro
PRO_PLUS: pro_plus';
        ALTER TABLE "usage_counters" ALTER COLUMN "resource_type" TYPE VARCHAR(11) USING "resource_type"::VARCHAR(11);
        COMMENT ON COLUMN "usage_counters"."resource_type" IS 'WORKFLOWS: workflows
RUNS: runs
LLM_CREDITS: llm_credits';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        COMMENT ON COLUMN "usage_counters"."resource_type" IS 'WORKFLOWS: workflows
RUNS: runs
CHAT_MESSAGES: chat_messages
LLM_CREDITS: llm_credits';
        ALTER TABLE "usage_counters" ALTER COLUMN "resource_type" TYPE VARCHAR(13) USING "resource_type"::VARCHAR(13);
        COMMENT ON COLUMN "billing_subscriptions"."tier" IS 'FREE: free
PRO: pro
PRO_PLUS: pro_plus
ULTRA: ultra';
        DROP TABLE IF EXISTS "overage_settings";
        DROP TABLE IF EXISTS "overage_usage_records";"""


MODELS_STATE = (
    "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"
)
