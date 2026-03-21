"""
test_pipeline.py
────────────────
Unit tests for chunker, BM25 retriever, hybrid fusion, and prompt templates.
Run: pytest tests/ -v
"""

import pytest
from langchain.schema import Document

from src.ingestion.chunker import ResearchPaperChunker
from src.ingestion.pdf_loader import DocumentPage
from src.retrieval.bm25_retriever import BM25Retriever
from src.pipeline.prompt_templates import RAG_PROMPT


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pages():
    return [
        DocumentPage(
            source="paper_a.pdf",
            page_number=1,
            text="Federated learning enables collaborative training without sharing raw data. "
                 "The FedAvg algorithm aggregates model weights from multiple clients. "
                 "Non-IID data distribution is a key challenge in federated settings.\n\n"
                 "Differential privacy adds noise to gradients to protect client data. "
                 "This provides formal privacy guarantees at the cost of model accuracy.",
            metadata={"source": "paper_a.pdf", "page": 1, "total_pages": 5},
        ),
        DocumentPage(
            source="paper_b.pdf",
            page_number=2,
            text="Transformer architectures rely on self-attention mechanisms. "
                 "BERT uses bidirectional encoders pre-trained on masked language modeling. "
                 "Fine-tuning on downstream tasks achieves state-of-the-art results.",
            metadata={"source": "paper_b.pdf", "page": 2, "total_pages": 8},
        ),
    ]


@pytest.fixture
def sample_chunks(sample_pages):
    chunker = ResearchPaperChunker(chunk_size=256, chunk_overlap=32)
    return chunker.chunk(sample_pages)


# ── Chunker Tests ─────────────────────────────────────────────────────────────

class TestChunker:
    def test_returns_documents(self, sample_chunks):
        assert len(sample_chunks) > 0
        assert all(isinstance(c, Document) for c in sample_chunks)

    def test_metadata_preserved(self, sample_chunks):
        for chunk in sample_chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_size_respected(self, sample_chunks):
        for chunk in sample_chunks:
            assert len(chunk.page_content) <= 300  # allow slight overflow

    def test_empty_pages_returns_empty(self):
        chunker = ResearchPaperChunker()
        result = chunker.chunk([])
        assert result == []


# ── BM25 Tests ────────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def test_retrieves_relevant_docs(self, sample_chunks):
        retriever = BM25Retriever.from_documents(sample_chunks, k=3)
        docs = retriever.get_relevant_documents("federated learning")
        assert len(docs) > 0
        # Top result should be from federated learning paper
        assert any("federated" in d.page_content.lower() for d in docs[:2])

    def test_returns_k_docs(self, sample_chunks):
        retriever = BM25Retriever.from_documents(sample_chunks, k=2)
        docs = retriever.get_relevant_documents("transformer attention")
        assert len(docs) <= 2

    def test_bm25_score_in_metadata(self, sample_chunks):
        retriever = BM25Retriever.from_documents(sample_chunks, k=3)
        docs = retriever.get_relevant_documents("privacy differential")
        for doc in docs:
            assert "bm25_score" in doc.metadata


# ── Prompt Template Tests ─────────────────────────────────────────────────────

class TestPromptTemplate:
    def test_prompt_formats_correctly(self):
        prompt = RAG_PROMPT.format(
            context="Federated learning is a distributed ML approach.",
            question="What is federated learning?",
        )
        assert "Federated learning" in prompt
        assert "What is federated learning?" in prompt
        assert "[INST]" not in prompt  # RAG_PROMPT is not the Mistral template

    def test_prompt_has_required_variables(self):
        assert "context" in RAG_PROMPT.input_variables
        assert "question" in RAG_PROMPT.input_variables
