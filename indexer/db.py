
from typing import Optional

import aiosqlite

from shared.logger import get_logger
from pathlib import Path

logger = get_logger("indexer.db")

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT.parent / "code_index.db"

DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

async def get_db_connection(db_path: Optional[str] = None) -> aiosqlite.Connection:
    path = db_path or DEFAULT_DB_PATH

    conn = await aiosqlite.connect(path)
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA synchronous=NORMAL;")
    await conn.execute("PRAGMA foreign_keys=ON;")
    await conn.commit()
    return conn


async def init_db(conn: aiosqlite.Connection) -> None:
    # Core file table
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            mtime REAL,
            size INTEGER,
            hash TEXT
        );
        """
    )
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);")

    # Symbol table
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            qualname TEXT,
            type TEXT NOT NULL,
            lineno INTEGER,
            end_lineno INTEGER,
            col_offset INTEGER,
            end_col_offset INTEGER,
            docstring TEXT,
            parent_qualname TEXT,
            decorators TEXT,
            returns TEXT,
            args TEXT,
            FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
        );
        """
    )
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_qualname ON symbols(qualname);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);")

    # Imports table
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS imports (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            module TEXT,
            name TEXT,
            alias TEXT,
            lineno INTEGER,
            FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
        );
        """
    )
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file_id);")

    # References table (approximate call/usage references)
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refs (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            from_symbol_id INTEGER,
            to_symbol_name TEXT,
            to_symbol_qualname TEXT,
            call_type TEXT,
            lineno INTEGER,
            FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
            FOREIGN KEY(from_symbol_id) REFERENCES symbols(id) ON DELETE SET NULL
        );
        """
    )
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_file ON refs(file_id);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_to_name ON refs(to_symbol_name);")

    # Chunk table (for semantic + lexical chunking)
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            content TEXT NOT NULL,
            md5 TEXT,
            FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
        );
        """
    )
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);")

    # Embeddings table: vector as BLOB (row-major float32 array)
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );
        """
    )

    # Lexical FTS on entire files
    try:
        await conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS file_fts USING fts5(
                path,
                content,
                tokenize='porter unicode61 remove_diacritics 2'
            );
            """
        )
    except Exception as e:
        logger.warning(f"FTS5 not available for file_fts: {e}")

    # Lexical FTS on chunks for more targeted search/snippets
    try:
        await conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                content,
                path,
                start_line UNINDEXED,
                end_line UNINDEXED,
                tokenize='porter unicode61 remove_diacritics 2'
            );
            """
        )
    except Exception as e:
        logger.warning(f"FTS5 not available for chunk_fts: {e}")

    # Lightweight FTS for symbols text and docstrings
    try:
        await conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS symbol_fts USING fts5(
                name,
                qualname,
                docstring,
                path,
                tokenize='porter unicode61 remove_diacritics 2'
            );
            """
        )
    except Exception as e:
        logger.warning(f"FTS5 not available for symbol_fts: {e}")

    await conn.commit()


