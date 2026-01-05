from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "integration_resources" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "provider" VARCHAR(50) NOT NULL,
    "resource_type" VARCHAR(50) NOT NULL,
    "resource_id" VARCHAR(255) NOT NULL,
    "resource_key" VARCHAR(255),
    "name" VARCHAR(255),
    "resource_metadata" JSONB,
    "status" VARCHAR(20) NOT NULL DEFAULT 'active',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "oauth_connection_id" BIGINT NOT NULL REFERENCES "oauth_connections" ("id") ON DELETE CASCADE,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_integration_oauth_c_f67a73" UNIQUE ("oauth_connection_id", "resource_type", "resource_id")
);
CREATE INDEX IF NOT EXISTS "idx_integration_user_id_5ce07e" ON "integration_resources" ("user_id", "provider", "resource_type");
COMMENT ON TABLE "integration_resources" IS 'Persisted resource binding that hangs off an OAuth connection.';
        CREATE TABLE IF NOT EXISTS "integration_secrets" (
    "id" BIGSERIAL NOT NULL PRIMARY KEY,
    "provider" VARCHAR(50) NOT NULL,
    "secret_type" VARCHAR(50) NOT NULL,
    "name" VARCHAR(100) NOT NULL,
    "value_enc" TEXT NOT NULL,
    "value_fingerprint" VARCHAR(64),
    "metadata" JSONB,
    "expires_at" TIMESTAMPTZ,
    "status" VARCHAR(20) NOT NULL DEFAULT 'active',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "oauth_connection_id" BIGINT REFERENCES "oauth_connections" ("id") ON DELETE CASCADE,
    "resource_id" BIGINT REFERENCES "integration_resources" ("id") ON DELETE CASCADE,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_integration_oauth_c_965e89" UNIQUE ("oauth_connection_id", "name"),
    CONSTRAINT "uid_integration_resourc_953d72" UNIQUE ("resource_id", "name")
);
CREATE INDEX IF NOT EXISTS "idx_integration_user_id_edae4e" ON "integration_secrets" ("user_id", "provider", "secret_type");
COMMENT ON TABLE "integration_secrets" IS 'Generic vault for non-OAuth credentials tied to a connection or resource.';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "integration_resources";
        DROP TABLE IF EXISTS "integration_secrets";"""


MODELS_STATE = (
    "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"
    "H7BHN10Yz7BKW2H/AOKee7zunlx/8BWIp71A=="
)
