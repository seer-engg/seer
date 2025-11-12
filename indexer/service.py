import ast
import asyncio
import hashlib
import json
import sqlite3
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from shared.logger import get_logger
from shared.schema import SandboxContext
from e2b_code_interpreter import AsyncSandbox, CommandResult
from sandbox.base import get_sandbox

from .db import get_db_connection, init_db
from .embedding import get_embedder

if TYPE_CHECKING:
    import aiosqlite

logger = get_logger("indexer.service")


def _md5_text(text: str) -> str:
    m = hashlib.md5()
    m.update(text.encode("utf-8"))
    return m.hexdigest()


def _vec_to_blob(vec: np.ndarray) -> bytes:
    assert vec.dtype == np.float32
    return vec.tobytes(order="C")


def _blob_to_vec(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    return getattr(obj, attr, default)


@dataclass
class SymbolInfo:
    name: str
    qualname: str
    type: str
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int
    docstring: Optional[str]
    parent_qualname: Optional[str]
    decorators: Optional[List[str]]
    returns: Optional[str]
    args: Optional[Dict[str, Any]]


class CodeIndexService:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.conn: Optional["aiosqlite.Connection"] = None
        self.db_path: Optional[str] = db_path
        self.embedder = get_embedder()
        self.lock = asyncio.Lock()

    async def _ensure_db(self) -> None:
        if self.conn is None:
            logger.info(f"Initializing database connection at path: {self.db_path}")
            try:
                self.conn = await get_db_connection(self.db_path)  # type: ignore[assignment]
                await init_db(self.conn)  # type: ignore[arg-type]
                logger.info("Database connection and initialization successful")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}", exc_info=True)
                raise

    async def _get_sandbox(self, sandbox_context: SandboxContext) -> AsyncSandbox:
        logger.info(f"Getting sandbox with ID: {sandbox_context.sandbox_id}")
        try:
            sandbox = await get_sandbox(sandbox_context.sandbox_id)
            logger.info(f"Successfully retrieved sandbox: {sandbox_context.sandbox_id}")
            return sandbox
        except Exception as e:
            logger.error(f"Failed to get sandbox {sandbox_context.sandbox_id}: {e}", exc_info=True)
            raise

    async def _list_python_files(self, sbx: AsyncSandbox, repo_path: str) -> List[str]:
        logger.info(f"Listing Python files in repository: {repo_path}")
        # Use find with path excludes; avoid nested bash -lc and complex quoting
        cmd = (
            "find . -type f -name '*.py' "
            "! -path '*/.git/*' "
            "! -path '*/__pycache__/*' "
            "! -path '*/env/*' "
            "! -path '*/venv/*' "
            "! -path '*/.venv/*' "
            "! -path '*/site-packages/*' "
            "! -path '*/node_modules/*' "
            "-print"
        )
        try:
            res: CommandResult = await sbx.commands.run(cmd, cwd=repo_path)
            if res.exit_code != 0:
                logger.warning(f"Failed to list python files: {res.stderr}")
                return []
            files = []
            for line in res.stdout.splitlines():
                p = line.strip()
                if not p:
                    continue
                if p.startswith("./"):
                    p = p[2:]
                files.append(p)
            logger.info(f"Found {len(files)} Python files in repository")
            return files
        except Exception as e:
            logger.error(f"Error listing Python files: {e}", exc_info=True)
            return []

    async def _stat_file(self, sbx: AsyncSandbox, full_path: str) -> Tuple[Optional[float], Optional[int]]:
        # GNU stat
        cmd = f"stat -c '%Y %s' {full_path}"
        try:
            res: CommandResult = await sbx.commands.run(cmd, cwd="/")
            if res.exit_code != 0:
                logger.debug(f"Failed to stat file {full_path}: {res.stderr}")
                return None, None
            parts = res.stdout.strip().split()
            if len(parts) != 2:
                logger.debug(f"Unexpected stat output for {full_path}: {res.stdout}")
                return None, None
            try:
                mtime = float(parts[0])
                size = int(parts[1])
                return mtime, size
            except Exception as e:
                logger.debug(f"Failed to parse stat output for {full_path}: {e}")
                return None, None
        except Exception as e:
            logger.warning(f"Error running stat command for {full_path}: {e}")
            return None, None

    async def index_repository(self, sandbox_context: SandboxContext, root: str = ".") -> Dict[str, Any]:
        logger.info(f"Starting repository indexing for sandbox {sandbox_context.sandbox_id}")
        await self._ensure_db()
        async with self.lock:
            try:
                sbx = await self._get_sandbox(sandbox_context)
                repo_path = sandbox_context.working_directory
                logger.info(f"Repository path: {repo_path}")
                
                py_files = await self._list_python_files(sbx, repo_path)
                logger.info(f"Found {len(py_files)} Python files in repository")
                
                seen_paths = set()
                updated = 0
                inserted = 0
                removed = 0
                
                for idx, rel_path in enumerate(py_files, 1):
                    logger.info(f"Processing file {idx}/{len(py_files)}: {rel_path}")
                    seen_paths.add(rel_path)
                    full_path = f"{repo_path}/{rel_path}"
                    
                    mtime, size = await self._stat_file(sbx, full_path)
                    logger.debug(f"File stats for {rel_path}: mtime={mtime}, size={size}")
                    
                    try:
                        content = await sbx.files.read(full_path)
                        logger.debug(f"Successfully read {rel_path} ({len(content)} bytes)")
                    except Exception as e:
                        logger.warning(f"Failed reading {rel_path}: {e}", exc_info=True)
                        continue
                    
                    try:
                        changed = await self._upsert_file(rel_path, content, mtime, size)
                        logger.info(f"Upserted file {rel_path}: {changed}")
                        if changed == "inserted":
                            inserted += 1
                        elif changed == "updated":
                            updated += 1
                    except Exception as e:
                        logger.error(f"Failed to upsert file {rel_path}: {e}", exc_info=True)
                        continue
                
                logger.info("Removing deleted files from index")
                removed = await self._remove_deleted(seen_paths)
                logger.info(f"Removed {removed} deleted files from index")
                
                result = {"inserted": inserted, "updated": updated, "removed": removed, "files_indexed": len(seen_paths)}
                logger.info(f"Repository indexing complete: {result}")
                return result
            except Exception as e:
                logger.error(f"Error during repository indexing: {e}", exc_info=True)
                raise

    async def _remove_deleted(self, seen_paths: set) -> int:
        assert self.conn is not None
        removed = 0
        try:
            async with self.conn.execute("SELECT id, path FROM files;") as cur:
                rows = await cur.fetchall()
            logger.info(f"Checking {len(rows)} existing files for deletion")
            for file_id, path in rows:
                if path not in seen_paths:
                    logger.info(f"Removing deleted file from index: {path}")
                    await self.conn.execute("DELETE FROM files WHERE id=?", (file_id,))
                    removed += 1
            await self.conn.commit()
            logger.info(f"Removed {removed} deleted files")
        except Exception as e:
            logger.error(f"Error removing deleted files: {e}", exc_info=True)
            raise
        return removed

    async def _upsert_file(self, rel_path: str, content: str, mtime: Optional[float], size: Optional[int]) -> str:
        logger.debug(f"Upserting file: {rel_path}")
        file_hash = _md5_text(content)
        assert self.conn is not None
        try:
            async with self.conn.execute("SELECT id, hash FROM files WHERE path=?", (rel_path,)) as cur:
                row = await cur.fetchone()
            if row:
                file_id, old_hash = row
                if old_hash == file_hash:
                    logger.debug(f"File {rel_path} is unchanged (hash match)")
                    return "unchanged"
                # update file row
                logger.info(f"Updating existing file: {rel_path}")
                await self.conn.execute(
                    "UPDATE files SET mtime=?, size=?, hash=? WHERE id=?",
                    (mtime, size, file_hash, file_id),
                )
                # purge dependent rows
                logger.debug(f"Purging dependent rows for file_id={file_id}")
                await self.conn.execute("DELETE FROM symbols WHERE file_id=?", (file_id,))
                await self.conn.execute("DELETE FROM imports WHERE file_id=?", (file_id,))
                await self.conn.execute("DELETE FROM refs WHERE file_id=?", (file_id,))
                await self.conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
                # FTS cleanup
                await self.conn.execute("DELETE FROM file_fts WHERE path=?", (rel_path,))
                await self.conn.execute("DELETE FROM chunk_fts WHERE path=?", (rel_path,))
                await self._index_file_contents(file_id, rel_path, content)
                await self.conn.commit()
                logger.info(f"File {rel_path} updated successfully")
                return "updated"
            else:
                logger.info(f"Inserting new file: {rel_path}")
                cur = await self.conn.execute(
                    "INSERT INTO files(path, mtime, size, hash) VALUES(?, ?, ?, ?)",
                    (rel_path, mtime, size, file_hash),
                )
                file_id = cur.lastrowid
                await cur.close()
                await self._index_file_contents(file_id, rel_path, content)
                await self.conn.commit()
                logger.info(f"File {rel_path} inserted successfully with file_id={file_id}")
                return "inserted"
        except Exception as e:
            logger.error(f"Error upserting file {rel_path}: {e}", exc_info=True)
            raise

    async def _index_file_contents(self, file_id: int, rel_path: str, content: str) -> None:
        logger.debug(f"Indexing file contents for {rel_path} (file_id={file_id})")
        assert self.conn is not None
        try:
            # File-level FTS
            await self.conn.execute("INSERT INTO file_fts(path, content) VALUES(?, ?)", (rel_path, content))
            logger.debug(f"Inserted file-level FTS for {rel_path}")

            # Parse AST for symbols/imports/references + chunk into symbol-based chunks
            try:
                tree = ast.parse(content)
                logger.debug(f"Successfully parsed AST for {rel_path}")
            except SyntaxError as e:
                logger.warning(f"Syntax error parsing {rel_path}: {e}")
                tree = None

            symbols: List[SymbolInfo] = []
            imports: List[Tuple[Optional[str], Optional[str], Optional[str], int]] = []
            references: List[Tuple[Optional[int], Optional[str], Optional[str], Optional[str], int]] = []
            chunks: List[Tuple[int, int, str]] = []

            lines = content.splitlines()

            def extract_symbol_info(node: ast.AST, parent_qual: Optional[str]) -> Optional[SymbolInfo]:
                if isinstance(node, ast.ClassDef):
                    qual = f"{parent_qual}.{node.name}" if parent_qual else node.name
                    doc = ast.get_docstring(node)
                    return SymbolInfo(
                        name=node.name,
                        qualname=qual,
                        type="class",
                        lineno=node.lineno,
                        end_lineno=_safe_get(node, "end_lineno", node.lineno),
                        col_offset=node.col_offset,
                        end_col_offset=_safe_get(node, "end_col_offset", 0),
                        docstring=doc,
                        parent_qualname=parent_qual,
                        decorators=[ast.unparse(d) for d in node.decorator_list] if hasattr(ast, "unparse") else None,
                        returns=None,
                        args=None,
                    )
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    qual = f"{parent_qual}.{node.name}" if parent_qual else node.name
                    doc = ast.get_docstring(node)
                    args = {
                        "args": [a.arg for a in node.args.args],
                        "vararg": node.args.vararg.arg if node.args.vararg else None,
                        "kwonlyargs": [a.arg for a in node.args.kwonlyargs],
                        "kwarg": node.args.kwarg.arg if node.args.kwarg else None,
                        "defaults": len(node.args.defaults) if node.args.defaults else 0,
                    }
                    returns = ast.unparse(node.returns) if hasattr(node, "returns") and node.returns is not None and hasattr(ast, "unparse") else None
                    return SymbolInfo(
                        name=node.name,
                        qualname=qual,
                        type="method" if parent_qual else "function",
                        lineno=node.lineno,
                        end_lineno=_safe_get(node, "end_lineno", node.lineno),
                        col_offset=node.col_offset,
                        end_col_offset=_safe_get(node, "end_col_offset", 0),
                        docstring=doc,
                        parent_qualname=parent_qual,
                        decorators=[ast.unparse(d) for d in node.decorator_list] if hasattr(ast, "unparse") else None,
                        returns=returns,
                        args=args,
                    )
                return None

            def walk(node: ast.AST, parent_qual: Optional[str]) -> None:
                # Imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((None, alias.name, alias.asname, node.lineno))
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                    for alias in node.names:
                        imports.append((module, alias.name, alias.asname, node.lineno))

                sym = extract_symbol_info(node, parent_qual)
                if sym:
                    symbols.append(sym)
                    # Create a chunk for this symbol's body if line info available
                    start = max(1, sym.lineno)
                    end = max(start, sym.end_lineno or start)
                    snippet = "\n".join(lines[start - 1 : end])
                    chunks.append((start, end, snippet))
                    # Recurse with new parent
                    new_parent = sym.qualname
                    for child in ast.iter_child_nodes(node):
                        walk(child, new_parent)
                    return

                # Collect call references (approximate)
                if isinstance(node, ast.Call):
                    callee = node.func
                    callee_str = None
                    if isinstance(callee, ast.Name):
                        callee_str = callee.id
                    elif isinstance(callee, ast.Attribute):
                        parts: List[str] = []
                        cur = callee
                        while isinstance(cur, ast.Attribute):
                            parts.append(cur.attr)
                            cur = cur.value
                        if isinstance(cur, ast.Name):
                            parts.append(cur.id)
                            parts.reverse()
                            callee_str = ".".join(parts)
                    if callee_str:
                        references.append((None, callee_str, None, "call", node.lineno))

                for child in ast.iter_child_nodes(node):
                    walk(child, parent_qual)

            if tree:
                walk(tree, None)
                logger.debug(f"Extracted {len(symbols)} symbols, {len(imports)} imports, {len(references)} references, {len(chunks)} chunks from {rel_path}")

            # Top-level chunk if no symbols captured content
            if not chunks:
                if len(lines) > 0:
                    chunks.append((1, len(lines), content))
                    logger.debug(f"Created top-level chunk for {rel_path}")

            # Insert symbols
            for sym in symbols:
                try:
                    await self.conn.execute(
                        """
                        INSERT INTO symbols(
                            file_id, name, qualname, type, lineno, end_lineno, col_offset, end_col_offset,
                            docstring, parent_qualname, decorators, returns, args
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            file_id,
                            sym.name,
                            sym.qualname,
                            sym.type,
                            sym.lineno,
                            sym.end_lineno,
                            sym.col_offset,
                            sym.end_col_offset,
                            sym.docstring,
                            sym.parent_qualname,
                            json.dumps(sym.decorators) if sym.decorators else None,
                            sym.returns,
                            json.dumps(sym.args) if sym.args else None,
                        ),
                    )
                except Exception as e:
                    logger.error(f"Failed to insert symbol {sym.qualname} for {rel_path}: {e}")
            logger.debug(f"Inserted {len(symbols)} symbols for {rel_path}")
            
            # Insert imports
            for module, name, alias, lineno in imports:
                try:
                    await self.conn.execute(
                        "INSERT INTO imports(file_id, module, name, alias, lineno) VALUES (?, ?, ?, ?, ?)",
                        (file_id, module, name, alias, lineno),
                    )
                except Exception as e:
                    logger.error(f"Failed to insert import {module}.{name} for {rel_path}: {e}")
            logger.debug(f"Inserted {len(imports)} imports for {rel_path}")

            # Insert references
            for from_sym_id, to_name, to_qual, call_type, lineno in references:
                try:
                    await self.conn.execute(
                        """
                        INSERT INTO refs(file_id, from_symbol_id, to_symbol_name, to_symbol_qualname, call_type, lineno)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (file_id, from_sym_id, to_name, to_qual, call_type, lineno),
                    )
                except Exception as e:
                    logger.error(f"Failed to insert reference to {to_name} for {rel_path}: {e}")
            logger.debug(f"Inserted {len(references)} references for {rel_path}")

            # Insert chunks + embeddings + chunk_fts
            contents = [c[2] for c in chunks]
            if contents:
                try:
                    vecs = await self.embedder.aencode(contents)
                    logger.debug(f"Generated {len(vecs)} embeddings for {rel_path}")
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for {rel_path}: {e}", exc_info=True)
                    vecs = np.zeros((0, self.embedder.dim), dtype=np.float32)
            else:
                vecs = np.zeros((0, self.embedder.dim), dtype=np.float32)
            
            for i, (start, end, text) in enumerate(chunks):
                try:
                    cur = await self.conn.execute(
                        "INSERT INTO chunks(file_id, start_line, end_line, content, md5) VALUES (?, ?, ?, ?, ?)",
                        (file_id, start, end, text, _md5_text(text)),
                    )
                    chunk_id = cur.lastrowid
                    await cur.close()
                    await self.conn.execute(
                        "INSERT INTO chunk_fts(content, path, start_line, end_line) VALUES (?, ?, ?, ?)",
                        (text, rel_path, start, end),
                    )
                    if vecs.shape[0] > i:
                        await self.conn.execute(
                            "INSERT OR REPLACE INTO embeddings(chunk_id, vector, dim) VALUES (?, ?, ?)",
                            (chunk_id, _vec_to_blob(vecs[i].astype(np.float32)), int(vecs.shape[1])),
                        )
                except Exception as e:
                    logger.error(f"Failed to insert chunk {i} for {rel_path}: {e}")
            logger.debug(f"Inserted {len(chunks)} chunks for {rel_path}")

            # Populate symbol_fts
            for sym in symbols:
                try:
                    await self.conn.execute(
                        "INSERT INTO symbol_fts(name, qualname, docstring, path) VALUES (?, ?, ?, ?)",
                        (sym.name, sym.qualname, sym.docstring or "", rel_path),
                    )
                except Exception as e:
                    logger.error(f"Failed to insert symbol FTS for {sym.qualname} in {rel_path}: {e}")
            logger.debug(f"Inserted symbol FTS entries for {rel_path}")
            
            logger.info(f"Successfully indexed file contents for {rel_path}")
        except Exception as e:
            logger.error(f"Error indexing file contents for {rel_path}: {e}", exc_info=True)
            raise

    async def update_files(self, sandbox_context: SandboxContext, relative_paths: Sequence[str]) -> Dict[str, Any]:
        logger.info(f"Updating {len(relative_paths)} files in index")
        await self._ensure_db()
        async with self.lock:
            try:
                sbx = await self._get_sandbox(sandbox_context)
                repo_path = sandbox_context.working_directory
                updated = 0
                inserted = 0
                for rel_path in relative_paths:
                    if not rel_path.endswith(".py"):
                        logger.debug(f"Skipping non-Python file: {rel_path}")
                        continue
                    full_path = f"{repo_path}/{rel_path}" if not rel_path.startswith("/") else rel_path
                    logger.info(f"Updating file: {rel_path}")
                    mtime, size = await self._stat_file(sbx, full_path)
                    try:
                        content = await sbx.files.read(full_path)
                    except Exception as e:
                        # File might be deleted
                        logger.info(f"File {rel_path} not found, removing from index: {e}")
                        await self._delete_by_path(rel_path)
                        continue
                    try:
                        status = await self._upsert_file(rel_path, content, mtime, size)
                        if status == "updated":
                            updated += 1
                        elif status == "inserted":
                            inserted += 1
                    except Exception as e:
                        logger.error(f"Failed to update file {rel_path}: {e}", exc_info=True)
                        continue
                result = {"inserted": inserted, "updated": updated}
                logger.info(f"File update complete: {result}")
                return result
            except Exception as e:
                logger.error(f"Error updating files: {e}", exc_info=True)
                raise

    async def _delete_by_path(self, rel_path: str) -> None:
        assert self.conn is not None
        try:
            logger.info(f"Deleting file from index: {rel_path}")
            await self.conn.execute("DELETE FROM files WHERE path=?", (rel_path,))
            await self.conn.commit()
            logger.info(f"Successfully deleted {rel_path} from index")
        except Exception as e:
            logger.error(f"Error deleting file {rel_path}: {e}", exc_info=True)
            raise

    # Query APIs
    async def search_code_lexical(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        logger.debug(f"Performing lexical search for query: '{query}' (k={k})")
        await self._ensure_db()
        assert self.conn is not None
        try:
            async with self.conn.execute(
                """
                SELECT path, start_line, end_line, snippet(chunk_fts, 0, '[[' , ']]', ' … ', 10)
                FROM chunk_fts
                WHERE chunk_fts MATCH ?
                LIMIT ?
                """,
                (query, k),
            ) as cur:
                rows = await cur.fetchall()
            logger.debug(f"Lexical search returned {len(rows)} results from chunk_fts")
        except sqlite3.OperationalError as e:
            logger.warning(f"chunk_fts search failed, falling back to file_fts: {e}")
            # fallback to file_fts
            try:
                async with self.conn.execute(
                    """
                    SELECT path, 1 as start_line, 1 as end_line, snippet(file_fts, 1, '[[' , ']]', ' … ', 10)
                    FROM file_fts
                    WHERE file_fts MATCH ?
                    LIMIT ?
                    """,
                    (query, k),
                ) as cur:
                    rows = await cur.fetchall()
                logger.debug(f"Lexical search returned {len(rows)} results from file_fts")
            except sqlite3.OperationalError as e2:
                logger.warning(f"file_fts search failed, falling back to LIKE: {e2}")
                # Fallback to LIKE over chunks table
                async with self.conn.execute(
                    """
                    SELECT files.path, MIN(chunks.start_line) as start_line, MAX(chunks.end_line) as end_line, substr(chunks.content, 1, 240)
                    FROM chunks 
                    JOIN files ON files.id = chunks.file_id
                    WHERE chunks.content LIKE ?
                    GROUP BY files.path
                    LIMIT ?
                    """,
                    (f"%{query}%", k),
                ) as cur:
                    rows = await cur.fetchall()
                logger.debug(f"Lexical search returned {len(rows)} results from LIKE fallback")
        results: List[Dict[str, Any]] = []
        for path, start, end, snip in rows:
            results.append({"path": path, "start_line": int(start), "end_line": int(end), "snippet": snip})
        logger.info(f"Lexical search complete: {len(results)} results")
        return results

    async def search_symbols(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        logger.debug(f"Searching symbols for query: '{query}' (k={k})")
        await self._ensure_db()
        assert self.conn is not None
        # Use FTS if possible, else fallback
        try:
            async with self.conn.execute(
                """
                SELECT path, name, qualname
                FROM symbol_fts
                WHERE symbol_fts MATCH ?
                LIMIT ?
                """,
                (query, k),
            ) as cur:
                rows = await cur.fetchall()
            if rows:
                logger.debug(f"Symbol FTS search returned {len(rows)} results")
                # enrich with line numbers
                out: List[Dict[str, Any]] = []
                for path, name, qual in rows:
                    async with self.conn.execute(
                        "SELECT lineno, end_lineno, type FROM symbols JOIN files ON files.id = symbols.file_id WHERE files.path=? AND symbols.qualname=?",
                        (path, qual),
                    ) as c2:
                        r2 = await c2.fetchone()
                    if r2:
                        out.append(
                            {"path": path, "name": name, "qualname": qual, "lineno": r2[0], "end_lineno": r2[1], "type": r2[2]}
                        )
                if out:
                    logger.info(f"Symbol search complete: {len(out)} results")
                    return out
        except sqlite3.OperationalError as e:
            logger.warning(f"Symbol FTS search failed, falling back to LIKE: {e}")
        # Fallback LIKE search
        async with self.conn.execute(
            """
            SELECT files.path, symbols.name, symbols.qualname, symbols.lineno, symbols.end_lineno, symbols.type
            FROM symbols JOIN files ON files.id = symbols.file_id
            WHERE symbols.name LIKE ? OR symbols.qualname LIKE ?
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", k),
        ) as cur:
            rows = await cur.fetchall()
        logger.info(f"Symbol search complete (LIKE fallback): {len(rows)} results")
        return [
            {"path": path, "name": name, "qualname": qual, "lineno": lineno, "end_lineno": end_lineno, "type": typ}
            for path, name, qual, lineno, end_lineno, typ in rows
        ]

    async def get_symbol_definition(self, qualname: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"Getting symbol definition for: {qualname}")
        await self._ensure_db()
        assert self.conn is not None
        try:
            async with self.conn.execute(
                """
                SELECT files.path, symbols.name, symbols.type, symbols.lineno, symbols.end_lineno
                FROM symbols JOIN files ON files.id = symbols.file_id
                WHERE symbols.qualname=?
                """,
                (qualname,),
            ) as cur:
                row = await cur.fetchone()
            if not row:
                logger.debug(f"No definition found for symbol: {qualname}")
                return None
            logger.debug(f"Found definition for symbol: {qualname}")
            return {"path": row[0], "name": row[1], "type": row[2], "lineno": row[3], "end_lineno": row[4]}
        except Exception as e:
            logger.error(f"Error getting symbol definition for {qualname}: {e}", exc_info=True)
            return None

    async def find_usages(self, name_or_qualname: str, k: int = 50) -> List[Dict[str, Any]]:
        logger.debug(f"Finding usages for: {name_or_qualname} (k={k})")
        await self._ensure_db()
        assert self.conn is not None
        try:
            async with self.conn.execute(
                """
                SELECT files.path, refs.lineno, refs.call_type
                FROM refs
                JOIN files ON files.id = refs.file_id
                WHERE refs.to_symbol_qualname=? OR refs.to_symbol_name=? 
                LIMIT ?
                """,
                (name_or_qualname, name_or_qualname, k),
            ) as cur:
                rows = await cur.fetchall()
            logger.info(f"Found {len(rows)} usages for: {name_or_qualname}")
            return [{"path": path, "lineno": lineno, "call_type": call_type} for path, lineno, call_type in rows]
        except Exception as e:
            logger.error(f"Error finding usages for {name_or_qualname}: {e}", exc_info=True)
            return []

    async def semantic_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        logger.debug(f"Performing semantic search for query: '{query}' (k={k})")
        await self._ensure_db()
        try:
            qvec = (await self.embedder.aencode([query])).astype(np.float32)[0]
            qnorm = np.linalg.norm(qvec)
            if qnorm > 0:
                qvec = qvec / qnorm
            logger.debug(f"Generated query embedding with dimension {qvec.shape[0]}")
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
            return []
        
        assert self.conn is not None
        try:
            async with self.conn.execute(
                "SELECT embeddings.chunk_id, embeddings.vector, embeddings.dim, chunks.start_line, chunks.end_line, files.path FROM embeddings JOIN chunks ON chunks.id=embeddings.chunk_id JOIN files ON files.id = chunks.file_id"
            ) as cur:
                rows = await cur.fetchall()
            logger.debug(f"Retrieved {len(rows)} embeddings for semantic search")
            
            scores: List[Tuple[float, Dict[str, Any]]] = []
            skipped = 0
            for chunk_id, blob, dim, start, end, path in rows:
                dim = int(dim)
                vec = _blob_to_vec(blob, dim)
                # Skip corrupted or incompatible vectors
                if vec.shape[0] != dim or qvec.shape[0] != dim:
                    skipped += 1
                    continue
                # vectors are stored normalized
                score = float(np.dot(qvec, vec))
                scores.append((score, {"path": path, "start_line": int(start), "end_line": int(end), "score": score}))
            
            if skipped > 0:
                logger.warning(f"Skipped {skipped} incompatible embeddings during semantic search")
            
            scores.sort(key=lambda x: x[0], reverse=True)
            results = [item for _, item in scores[:k]]
            logger.info(f"Semantic search complete: {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error during semantic search: {e}", exc_info=True)
            return []

    async def get_file_region(self, path: str, start_line: int, end_line: int) -> Optional[str]:
        logger.debug(f"Getting file region: {path} lines {start_line}-{end_line}")
        await self._ensure_db()
        assert self.conn is not None
        try:
            async with self.conn.execute(
                """
                SELECT content FROM chunks 
                JOIN files ON files.id = chunks.file_id 
                WHERE files.path=? AND chunks.start_line=? AND chunks.end_line=?
                """,
                (path, start_line, end_line),
            ) as cur:
                row = await cur.fetchone()
            if row:
                logger.debug(f"Found file region for {path} lines {start_line}-{end_line}")
                return row[0]
            logger.debug(f"No file region found for {path} lines {start_line}-{end_line}")
            # Raw fallback by reading file_fts is not straightforward; this is best-effort
            return None
        except Exception as e:
            logger.error(f"Error getting file region for {path}: {e}", exc_info=True)
            return None


_global_service: Optional[CodeIndexService] = None


def get_index_service() -> CodeIndexService:
    global _global_service
    if _global_service is None:
        logger.info("Creating global CodeIndexService instance")
        _global_service = CodeIndexService()
    return _global_service




async def index_codebase(sandbox_context: SandboxContext) -> str:
    """
    Build a fresh hybrid (lexical + structural + semantic) index of the repository in the current sandbox.
    This creates/updates a SQLite DB in the project root (outside the sandbox).
    """
    logger.info("index_codebase called")
    if not sandbox_context:
        logger.error("Runtime context not found - sandbox_context is None")
        raise ValueError("Runtime context not found. Make sure context is passed when invoking the agent.")
    try:
        service = get_index_service()
        logger.info(f"Indexing codebase for sandbox {sandbox_context.sandbox_id} in directory {sandbox_context.working_directory}")
        result = await service.index_repository(sandbox_context)
        success_msg = f"Index built. Files indexed: {result.get('files_indexed')}, inserted: {result.get('inserted')}, updated: {result.get('updated')}, removed: {result.get('removed')}"
        logger.info(f"Indexing complete: {success_msg}")
        return success_msg
    except Exception as e:
        error_msg = f"Error indexing codebase: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return error_msg
