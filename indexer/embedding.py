import hashlib
import os
from typing import Iterable, List, Optional

import numpy as np
import asyncio
from langchain_openai import OpenAIEmbeddings

from shared.logger import get_logger
import traceback
from shared.config import config
logger = get_logger("indexer.embedding")


class HashingEmbedder:
    def __init__(self, dim: int = 1536):
        self.dim = dim

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for tok in self._simple_tokenize(text):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.stack(vectors, axis=0) if vectors else np.zeros((0, self.dim), dtype=np.float32)

    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        out: List[str] = []
        buff: List[str] = []
        for ch in text:
            if ch.isalnum() or ch == "_":
                buff.append(ch.lower())
            else:
                if buff:
                    out.append("".join(buff))
                    buff = []
        if buff:
            out.append("".join(buff))
        return out


class OpenAIEmbedder:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        # Default to text-embedding-3-small; allow override to -large via env or param
        self.model = model or config.embedding_model
        self.api_key = api_key or config.openai_api_key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI embeddings")
        self.client = OpenAIEmbeddings(model=self.model, api_key=self.api_key)
        # Known dims for current OpenAI models
        if self.model.endswith("3-small"):
            self.dim = 1536
        elif self.model.endswith("3-large"):
            self.dim = 3072
        else:
            # Best-effort default; will be corrected at runtime if needed
            self.dim = 1536

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        # Synchronous path (may trigger blocking warnings); prefer using aencode in async contexts
        texts_list = list(texts)
        vectors: List[List[float]] = []
        batch_size = config.embedding_batch_size
        for i in range(0, len(texts_list), batch_size):
            chunk = texts_list[i : i + batch_size]
            try:
                chunk_vecs = self.client.embed_documents(chunk)
            except Exception:
                logger.error(f"OpenAI embeddings failed: {traceback.format_exc()}")
                raise
            vectors.extend(chunk_vecs)
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        if arr.shape[1] != self.dim:
            self.dim = int(arr.shape[1])
        return arr

    async def aencode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        vectors: List[List[float]] = []
        batch_size = config.embedding_batch_size
        for i in range(0, len(texts_list), batch_size):
            chunk = texts_list[i : i + batch_size]
            try:
                chunk_vecs = await asyncio.to_thread(self.client.embed_documents, chunk)
            except Exception:
                logger.error(f"OpenAI embeddings failed (async): {traceback.format_exc()}")
                raise
            vectors.extend(chunk_vecs)
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        if arr.shape[1] != self.dim:
            self.dim = int(arr.shape[1])
        return arr

    async def aencode_query(self, text: str) -> np.ndarray:
        try:
            vec = await asyncio.to_thread(self.client.embed_query, text)
        except Exception:
            logger.error(f"OpenAI query embedding failed (async): {traceback.format_exc()}")
            raise
        arr = np.array([vec], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        if arr.shape[1] != self.dim:
            self.dim = int(arr.shape[1])
        return arr


class Embedder:
    def __init__(self):
        self._impl = None
        self.dim = 1536
        self._init_impl()

    def _init_impl(self) -> None:
        try:
            self._impl = OpenAIEmbedder()
            self.dim = self._impl.dim
            logger.info(f"Using OpenAI embeddings model: {self._impl.model} (dim={self.dim})")
        except Exception as e:
            logger.warning(f"OpenAI embeddings unavailable, falling back to hashing embedder: {e}")
            self._impl = HashingEmbedder(dim=self.dim)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        # Keep sync path for non-async callers, but prefer aencode to avoid blocking issues
        try:
            return self._impl.encode(texts)
        except Exception as e:
            logger.warning(f"Primary embedder failed, using hashing fallback: {e}")
            fallback = HashingEmbedder(dim=self.dim)
            return fallback.encode(texts)

    async def aencode(self, texts: Iterable[str]) -> np.ndarray:
        try:
            if hasattr(self._impl, "aencode"):
                return await self._impl.aencode(texts)  # type: ignore[attr-defined]
            return HashingEmbedder(dim=self.dim).encode(texts)
        except Exception as e:
            logger.warning(f"Primary embedder failed async, using hashing fallback: {e}")
            return HashingEmbedder(dim=self.dim).encode(texts)

    async def aencode_query(self, text: str) -> np.ndarray:
        try:
            if hasattr(self._impl, "aencode_query"):
                return await self._impl.aencode_query(text)  # type: ignore[attr-defined]
            return HashingEmbedder(dim=self.dim).encode([text])
        except Exception as e:
            logger.warning(f"Primary query embedder failed async, using hashing fallback: {e}")
            return HashingEmbedder(dim=self.dim).encode([text])


_global_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = Embedder()
    return _global_embedder


