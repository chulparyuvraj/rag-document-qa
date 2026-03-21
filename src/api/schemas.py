"""
schemas.py — Pydantic request/response models for the FastAPI endpoints.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, example="What is federated learning?")
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = Field(default=True)


class SourceDocument(BaseModel):
    content: str
    source: str
    page: int
    rrf_score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = []
    latency_ms: float


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    total_chunks: int
