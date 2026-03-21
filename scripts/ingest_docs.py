"""
ingest_docs.py
──────────────
Standalone script to build the FAISS index from a PDF directory.
Run this once before starting the API server.

Usage:
    python scripts/ingest_docs.py --pdf_dir data/papers --index_path indexes/faiss
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.rag_chain import RAGPipeline
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from PDFs")
    parser.add_argument("--pdf_dir", default="data/papers")
    parser.add_argument("--index_path", default="indexes/faiss")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    rag = RAGPipeline(index_path=args.index_path)
    rag.build_index(
        pdf_dir=args.pdf_dir,
        save_index=True,
        force_rebuild=args.force_rebuild,
    )
    print(f"\n✅ Index built: {len(rag._chunks)} chunks indexed")
    print(f"   Saved to: {args.index_path}")


if __name__ == "__main__":
    main()
