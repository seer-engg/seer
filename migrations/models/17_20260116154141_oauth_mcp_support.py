from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "oauth_authorization_codes" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "code" VARCHAR(128) NOT NULL UNIQUE,
    "client_id" VARCHAR(255) NOT NULL,
    "redirect_uri" VARCHAR(2048) NOT NULL,
    "code_challenge" VARCHAR(128) NOT NULL,
    "code_challenge_method" VARCHAR(10) NOT NULL,
    "scope" VARCHAR(500),
    "expires_at" TIMESTAMPTZ NOT NULL,
    "used" BOOL NOT NULL DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_oauth_autho_code_637531" ON "oauth_authorization_codes" ("code");
COMMENT ON TABLE "oauth_authorization_codes" IS 'OAuth 2.1 authorization codes for MCP clients.';
        CREATE TABLE IF NOT EXISTS "oauth_refresh_tokens" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "token" VARCHAR(255) NOT NULL UNIQUE,
    "client_id" VARCHAR(255) NOT NULL,
    "scope" VARCHAR(500),
    "expires_at" TIMESTAMPTZ NOT NULL,
    "revoked" BOOL NOT NULL DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "last_used_at" TIMESTAMPTZ,
    "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_oauth_refre_token_03ed63" ON "oauth_refresh_tokens" ("token");
COMMENT ON TABLE "oauth_refresh_tokens" IS 'OAuth 2.1 refresh tokens for MCP clients.';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "oauth_authorization_codes";
        DROP TABLE IF EXISTS "oauth_refresh_tokens";"""


MODELS_STATE = (
    "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"
    "c="
)
