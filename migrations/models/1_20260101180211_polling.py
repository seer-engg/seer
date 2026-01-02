from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "trigger_events" ADD "event_hash" VARCHAR(255);
        ALTER TABLE "trigger_subscriptions" ADD "poll_backoff_seconds" INT NOT NULL DEFAULT 0;
        ALTER TABLE "trigger_subscriptions" ADD "poll_cursor_json" JSONB;
        ALTER TABLE "trigger_subscriptions" ADD "poll_lock_expires_at" TIMESTAMPTZ;
        ALTER TABLE "trigger_subscriptions" ADD "poll_status" VARCHAR(32) NOT NULL DEFAULT 'ok';
        ALTER TABLE "trigger_subscriptions" ADD "poll_lock_owner" VARCHAR(255);
        ALTER TABLE "trigger_subscriptions" ADD "poll_error_json" JSONB;
        ALTER TABLE "trigger_subscriptions" ADD "next_poll_at" TIMESTAMPTZ NOT NULL;
        ALTER TABLE "trigger_subscriptions" ADD "poll_interval_seconds" INT NOT NULL DEFAULT 60;
        COMMENT ON COLUMN "trigger_events"."event_hash" IS 'Deterministic hash used when provider_event_id is unavailable.';
COMMENT ON COLUMN "trigger_subscriptions"."next_poll_at" IS 'Next scheduled poll time (UTC).';
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_trigger_eve_trigger_bfc70c" ON "trigger_events" ("trigger_key", "provider_connection_id", "event_hash");"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "uid_trigger_eve_trigger_bfc70c";
        ALTER TABLE "trigger_events" DROP COLUMN "event_hash";
        ALTER TABLE "trigger_subscriptions" DROP COLUMN "poll_backoff_seconds";
        ALTER TABLE "trigger_subscriptions" DROP COLUMN "poll_cursor_json";
        ALTER TABLE "trigger_subscriptions" DROP COLUMN "poll_lock_expires_at";
        ALTER TABLE "trigger_subscriptions" DROP COLUMN "poll_status";
        ALTER TABLE "trigger_subscriptions" DROP COLUMN "poll_lock_owner";
        ALTER TABLE "trigger_subscriptions" DROP COLUMN "poll_error_json";
        ALTER TABLE "trigger_subscriptions" DROP COLUMN "next_poll_at";
        ALTER TABLE "trigger_subscriptions" DROP COLUMN "poll_interval_seconds";"""


MODELS_STATE = (
    "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"
)
