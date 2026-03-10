from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "stripe_customers" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "stripe_customer_id" VARCHAR(255) NOT NULL UNIQUE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by_user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "stripe_customers" IS 'Tracks Stripe customer ownership with full audit trail.';
        ALTER TABLE "billing_subscriptions" ADD "organization_id" INT UNIQUE;
        ALTER TABLE "billing_subscriptions" ALTER COLUMN "billing_profile_id" DROP NOT NULL;
        ALTER TABLE "overage_settings" ADD "organization_id" INT UNIQUE;
        ALTER TABLE "overage_settings" ALTER COLUMN "billing_profile_id" DROP NOT NULL;
        ALTER TABLE "organizations" ADD "payment_method_added_at" TIMESTAMPTZ;
        ALTER TABLE "organizations" ADD "stripe_customer_id" INT;
        ALTER TABLE "organizations" ADD "has_payment_method" BOOL NOT NULL DEFAULT False;
        ALTER TABLE "billing_subscriptions" ADD CONSTRAINT "fk_billing__organiza_30925138" FOREIGN KEY ("organization_id") REFERENCES "organizations" ("id") ON DELETE CASCADE;
        ALTER TABLE "overage_settings" ADD CONSTRAINT "fk_overage__organiza_33337fef" FOREIGN KEY ("organization_id") REFERENCES "organizations" ("id") ON DELETE CASCADE;
        ALTER TABLE "organizations" ADD CONSTRAINT "fk_organiza_stripe_c_6b32634d" FOREIGN KEY ("stripe_customer_id") REFERENCES "stripe_customers" ("id") ON DELETE SET NULL;

        -- Backfill billing_subscriptions.organization_id from billing_profiles
        -- Step 1: Use billing_profile.owner_organization_id if it exists
        UPDATE billing_subscriptions bs
        SET organization_id = bp.owner_organization_id
        FROM billing_profiles bp
        WHERE bs.billing_profile_id = bp.id
          AND bs.organization_id IS NULL
          AND bp.owner_organization_id IS NOT NULL;

        -- Step 2: For profiles without owner_organization_id, use the user's personal org
        UPDATE billing_subscriptions bs
        SET organization_id = o.id
        FROM billing_profiles bp
        JOIN organizations o ON o.owner_id = bp.owner_user_id AND o.type = 'personal'
        WHERE bs.billing_profile_id = bp.id
          AND bs.organization_id IS NULL
          AND bp.owner_organization_id IS NULL
          AND bp.owner_user_id IS NOT NULL;

        -- Backfill overage_settings.organization_id (same logic)
        UPDATE overage_settings os
        SET organization_id = bp.owner_organization_id
        FROM billing_profiles bp
        WHERE os.billing_profile_id = bp.id
          AND os.organization_id IS NULL
          AND bp.owner_organization_id IS NOT NULL;

        UPDATE overage_settings os
        SET organization_id = o.id
        FROM billing_profiles bp
        JOIN organizations o ON o.owner_id = bp.owner_user_id AND o.type = 'personal'
        WHERE os.billing_profile_id = bp.id
          AND os.organization_id IS NULL
          AND bp.owner_organization_id IS NULL
          AND bp.owner_user_id IS NOT NULL;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "billing_subscriptions" DROP CONSTRAINT IF EXISTS "fk_billing__organiza_30925138";
        ALTER TABLE "overage_settings" DROP CONSTRAINT IF EXISTS "fk_overage__organiza_33337fef";
        ALTER TABLE "organizations" DROP CONSTRAINT IF EXISTS "fk_organiza_stripe_c_6b32634d";
        ALTER TABLE "organizations" DROP COLUMN "payment_method_added_at";
        ALTER TABLE "organizations" DROP COLUMN "stripe_customer_id";
        ALTER TABLE "organizations" DROP COLUMN "has_payment_method";
        ALTER TABLE "overage_settings" DROP COLUMN "organization_id";
        ALTER TABLE "overage_settings" ALTER COLUMN "billing_profile_id" SET NOT NULL;
        ALTER TABLE "billing_subscriptions" DROP COLUMN "organization_id";
        ALTER TABLE "billing_subscriptions" ALTER COLUMN "billing_profile_id" SET NOT NULL;
        DROP TABLE IF EXISTS "stripe_customers";"""


MODELS_STATE = (
    "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"
)
