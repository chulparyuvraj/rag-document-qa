"""
embeddings.py
─────────────
Wraps HuggingFace sentence-transformers for dense embedding generation.
Model: all-MiniLM-L6-v2  (384-dim, fast, strong on scientific text)
Swap to 'allenai/specter2' for domain-specific research paper embeddings.
"""

import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


def get_embedding_model(
    model_name: str = None,
    device: str = "cuda",          # "cpu" fallback if no GPU
    normalize: bool = True,
) -> HuggingFaceEmbeddings:
    """
    Returns a LangChain-compatible HuggingFace embedding model.

    Args:
        model_name: HuggingFace model ID. Defaults to env var or MiniLM.
        device:     'cuda' for GPU, 'cpu' for CPU-only environments.
        normalize:  L2-normalize embeddings (required for cosine similarity).

    Returns:
        HuggingFaceEmbeddings instance ready for FAISS indexing.
    """
    model_name = model_name or os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    logger.info(f"Loading embedding model: {model_name} on {device}")

    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": normalize,
            "batch_size": 64,       # tune based on GPU VRAM
        },
    )

    logger.info("Embedding model loaded successfully")
    return model
