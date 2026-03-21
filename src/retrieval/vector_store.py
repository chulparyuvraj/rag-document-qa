"""
vector_store.py
───────────────
Builds, saves, and loads a FAISS vector index from document chunks.
Supports incremental updates (add new papers without rebuilding).
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


class FAISSVectorStore:
    """
    Manages the FAISS dense vector index for research paper chunks.

    Usage:
        store = FAISSVectorStore(embedding_model)
        store.build(chunks)          # first time
        store.save("indexes/faiss")
        store.load("indexes/faiss")  # subsequent runs
        docs = store.similarity_search("attention mechanism", k=5)
    """

    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        self.embedding_model = embedding_model
        self._index: Optional[FAISS] = None

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self, chunks: List[Document], batch_size: int = 256) -> None:
        """
        Create FAISS index from scratch.
        Batched to avoid OOM on large corpora.
        """
        logger.info(f"Building FAISS index from {len(chunks)} chunks...")

        # Build index with first batch, then add remaining
        self._index = FAISS.from_documents(
            chunks[:batch_size], self.embedding_model
        )
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self._index.add_documents(batch)
            logger.debug(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

        logger.info("FAISS index built successfully")

    def add_documents(self, new_chunks: List[Document]) -> None:
        """Incrementally add new paper chunks without rebuilding the index."""
        if self._index is None:
            raise RuntimeError("Index not initialized. Call build() first.")
        self._index.add_documents(new_chunks)
        logger.info(f"Added {len(new_chunks)} new chunks to existing index")

    # ── Persist ───────────────────────────────────────────────────────────

    def save(self, index_path: str) -> None:
        """Save FAISS index to disk."""
        Path(index_path).mkdir(parents=True, exist_ok=True)
        self._index.save_local(index_path)
        logger.info(f"FAISS index saved to {index_path}")

    def load(self, index_path: str) -> None:
        """Load FAISS index from disk."""
        self._index = FAISS.load_local(
            index_path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"FAISS index loaded from {index_path}")

    # ── Search ────────────────────────────────────────────────────────────

    def similarity_search(
        self, query: str, k: int = 5, score_threshold: float = 0.0
    ) -> List[Document]:
        """Dense similarity search — returns top-k chunks."""
        if self._index is None:
            raise RuntimeError("Index not initialized. Call build() or load().")
        return self._index.similarity_search(query, k=k)

    def as_retriever(self, k: int = 5):
        """Return a LangChain-compatible retriever object."""
        return self._index.as_retriever(search_kwargs={"k": k})
