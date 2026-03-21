
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
from loguru import logger
import re


class BM25Retriever(BaseRetriever):
    documents: List[Document]
    bm25: object
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(cls, documents: List[Document], k: int = 5) -> "BM25Retriever":
        tokenized_corpus = [cls._tokenize(doc.page_content) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built on {len(documents)} documents")
        return cls(documents=documents, bm25=bm25, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:self.k]
        results = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            doc.metadata["bm25_score"] = float(scores[idx])
            results.append(doc)
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
