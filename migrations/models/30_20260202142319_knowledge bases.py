from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Knowledge bases table
        CREATE TABLE IF NOT EXISTS "knowledge_bases" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "name" VARCHAR(255) NOT NULL,
            "description" TEXT,
            "embedding_model" VARCHAR(100) NOT NULL DEFAULT 'text-embedding-3-small',
            "embedding_dims" INT NOT NULL DEFAULT 1536,
            "chunk_size" INT NOT NULL DEFAULT 1000,
            "chunk_overlap" INT NOT NULL DEFAULT 200,
            "metadata" JSONB,
            "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "user_id" INT NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS "idx_knowledge_bases_user_id" ON "knowledge_bases" ("user_id");
        COMMENT ON TABLE "knowledge_bases" IS 'Knowledge base entity for storing document collections and their embeddings.';

        -- Knowledge documents table
        CREATE TABLE IF NOT EXISTS "knowledge_documents" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "name" VARCHAR(255) NOT NULL,
            "mime_type" VARCHAR(100) NOT NULL,
            "file_size" INT NOT NULL,
            "content_hash" VARCHAR(64) NOT NULL,
            "chunk_count" INT NOT NULL DEFAULT 0,
            "processing_status" VARCHAR(20) NOT NULL DEFAULT 'pending',
            "processing_error" TEXT,
            "metadata" JSONB,
            "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "knowledge_base_id" INT NOT NULL REFERENCES "knowledge_bases" ("id") ON DELETE CASCADE,
            CONSTRAINT "uid_knowledge_doc_kb_hash" UNIQUE ("knowledge_base_id", "content_hash")
        );
        CREATE INDEX IF NOT EXISTS "idx_knowledge_documents_kb_id" ON "knowledge_documents" ("knowledge_base_id");
        CREATE INDEX IF NOT EXISTS "idx_knowledge_documents_status" ON "knowledge_documents" ("processing_status");
        COMMENT ON TABLE "knowledge_documents" IS 'Document within a knowledge base.';

        -- Knowledge chunks table with pgvector embedding
        CREATE TABLE IF NOT EXISTS "knowledge_chunks" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "chunk_index" INT NOT NULL,
            "content" TEXT NOT NULL,
            "embedding" vector(1536),
            "metadata" JSONB,
            "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "document_id" INT NOT NULL REFERENCES "knowledge_documents" ("id") ON DELETE CASCADE,
            "knowledge_base_id" INT NOT NULL REFERENCES "knowledge_bases" ("id") ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS "idx_knowledge_chunks_doc_id" ON "knowledge_chunks" ("document_id");
        CREATE INDEX IF NOT EXISTS "idx_knowledge_chunks_kb_id" ON "knowledge_chunks" ("knowledge_base_id");
        COMMENT ON TABLE "knowledge_chunks" IS 'Text chunk with embedding for semantic search.';

        -- IVFFlat index for similarity search (cosine distance)
        -- Using 100 lists as a good default for medium-sized datasets
        CREATE INDEX IF NOT EXISTS "idx_chunk_embedding_ivfflat"
        ON "knowledge_chunks"
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

        """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "knowledge_bases";
        DROP TABLE IF EXISTS "knowledge_documents";
        DROP TABLE IF EXISTS "knowledge_chunks";"""


MODELS_STATE = (
    "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"
    "Nv8o22/MAbeSpaZLRlVJc63WR//j8aAQ+h"
)
