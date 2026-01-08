from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(_db: BaseDBAsyncClient) -> str:
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
    "oauth_connection_id" BIGINT REFERENCES "oauth_connections" ("id") ON DELETE CASCADE,
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


async def downgrade(_db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "integration_secrets";
        DROP TABLE IF EXISTS "integration_resources";"""


MODELS_STATE = (
    "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"
    "QP5M8j7BQl005z5Bpe2HvEPG+W5yevn+fyr/580="
)
