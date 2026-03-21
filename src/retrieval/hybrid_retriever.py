
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from loguru import logger

from .vector_store import FAISSVectorStore
from .bm25_retriever import BM25Retriever


class HybridRetriever(BaseRetriever):
    faiss_store: FAISSVectorStore
    bm25_retriever: BM25Retriever
    k: int = 5
    rrf_k: int = 60
    dense_weight: float = 0.6
    sparse_weight: float = 0.4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        fetch_k = self.k * 3
        dense_docs  = self.faiss_store.similarity_search(query, k=fetch_k)
        sparse_docs = self.bm25_retriever._get_relevant_documents(query)
        fused = self._rrf(dense_docs, sparse_docs)
        logger.debug(f"Hybrid: dense={len(dense_docs)}, sparse={len(sparse_docs)}, returning top {self.k}")
        return fused[:self.k]

    def _rrf(self, dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for rank, doc in enumerate(dense_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + self.dense_weight * (1.0 / (self.rrf_k + rank + 1))
            doc_map[key] = doc
        for rank, doc in enumerate(sparse_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))
            doc_map[key] = doc
        sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
        results = []
        for key in sorted_keys:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = round(scores[key], 6)
            results.append(doc)
        return results
