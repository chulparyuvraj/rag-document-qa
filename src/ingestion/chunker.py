"""
chunker.py
──────────
Splits extracted pages into overlapping chunks suitable for embedding.
Uses LangChain's RecursiveCharacterTextSplitter — best for research papers
because it respects paragraph → sentence → word hierarchy.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from .pdf_loader import DocumentPage


class ResearchPaperChunker:
    """
    Converts DocumentPages into LangChain Documents (chunks).

    Strategy for research papers:
      - chunk_size=512  tokens ≈ one paragraph / abstract section
      - chunk_overlap=64 ensures cross-boundary context is preserved
      - Separators: paragraph breaks → newlines → sentences → words

    Usage:
        chunker = ResearchPaperChunker()
        chunks  = chunker.chunk(pages)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, pages: List[DocumentPage]) -> List[Document]:
        """
        Convert raw pages → overlapping LangChain Document chunks.
        Each chunk carries full provenance metadata.
        """
        all_chunks: List[Document] = []

        for page in pages:
            splits = self.splitter.split_text(page.text)
            for i, split_text in enumerate(splits):
                all_chunks.append(Document(
                    page_content=split_text,
                    metadata={
                        **page.metadata,
                        "chunk_index": i,
                        "chunk_total": len(splits),
                    }
                ))

        logger.info(
            f"Chunked {len(pages)} pages → {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def chunk_single(self, text: str, metadata: dict = None) -> List[Document]:
        """Chunk a raw string directly (useful for quick tests)."""
        splits = self.splitter.split_text(text)
        return [
            Document(page_content=s, metadata=metadata or {})
            for s in splits
        ]
