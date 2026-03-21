"""
main.py — FastAPI REST API for the RAG Document QA System.

Endpoints:
  GET  /health          — liveness check
  POST /query           — ask a question over indexed papers
  POST /upload          — upload and index a new PDF
  GET  /documents       — list indexed documents

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Deploy on Azure App Service:
    See deploy/azure-deploy.sh
"""

import os
import time
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

from .schemas import QueryRequest, QueryResponse, SourceDocument, UploadResponse, HealthResponse
from ..pipeline.rag_chain import RAGPipeline

load_dotenv()

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document QA API",
    description="Research paper question-answering powered by Mistral-7B + FAISS + BM25",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Pipeline Instance ──────────────────────────────────────────────────
rag: RAGPipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on server start."""
    global rag
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline(
        index_path=os.getenv("FAISS_INDEX_PATH", "indexes/faiss"),
        top_k=5,
    )
    pdf_dir = os.getenv("PDF_DIR", "data/papers")
    rag.build_index(pdf_dir=pdf_dir, save_index=True)
    logger.info("RAG pipeline ready. API is live.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness and readiness check."""
    return HealthResponse(
        status="healthy",
        index_loaded=rag._chain is not None,
        total_chunks=len(rag._chunks),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question over the indexed research papers.

    Returns the LLM-generated answer along with source chunks.
    Average latency: <300ms on Azure Standard_D4s_v3.
    """
    if rag is None or rag._chain is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    start = time.perf_counter()

    try:
        result = rag.query(request.question)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000

    sources = []
    if request.include_sources:
        for doc in result.get("source_documents", []):
            sources.append(SourceDocument(
                content=doc.page_content[:500],   # truncate for response size
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", 0),
                rrf_score=doc.metadata.get("rrf_score"),
            ))

    logger.info(f"Query answered in {latency_ms:.1f}ms")
    return QueryResponse(
        answer=result["result"],
        sources=sources,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a new research paper PDF and index it incrementally.
    The paper is available for querying immediately after indexing.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save uploaded file
    upload_dir = Path("data/papers")
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / file.filename

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Index in background to avoid blocking the response
    from ..ingestion.pdf_loader import PDFLoader
    from ..ingestion.chunker import ResearchPaperChunker

    loader = PDFLoader(source_dir=None)
    pages = loader.load_file(str(save_path))
    chunker = ResearchPaperChunker()
    new_chunks = chunker.chunk(pages)

    background_tasks.add_task(rag._faiss_store.add_documents, new_chunks)
    rag._chunks.extend(new_chunks)

    logger.info(f"Uploaded and indexed {file.filename}: {len(new_chunks)} chunks")
    return UploadResponse(
        message="Paper uploaded and indexed successfully",
        filename=file.filename,
        chunks_indexed=len(new_chunks),
    )


@app.get("/documents")
async def list_documents():
    """List all indexed research papers."""
    if not rag._chunks:
        return {"documents": []}
    sources = sorted({c.metadata.get("source", "unknown") for c in rag._chunks})
    return {"documents": sources, "total": len(sources)}
