"""
pdf_loader.py
─────────────
Loads research paper PDFs from a local directory or Azure Blob Storage,
extracts raw text using PyMuPDF (fast) with PyPDF fallback.
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from loguru import logger
from tqdm import tqdm


@dataclass
class DocumentPage:
    """Represents a single page extracted from a PDF."""
    source: str          # filename
    page_number: int
    text: str
    metadata: dict = field(default_factory=dict)


class PDFLoader:
    """
    Loads PDFs from a local folder or Azure Blob Storage.

    Usage:
        loader = PDFLoader(source_dir="data/papers")
        pages  = loader.load_all()
    """

    def __init__(
        self,
        source_dir: Optional[str] = None,
        azure_container: Optional[str] = None,
        min_page_chars: int = 50,       # skip near-blank pages
    ):
        self.source_dir = Path(source_dir) if source_dir else None
        self.azure_container = azure_container
        self.min_page_chars = min_page_chars

    # ── Public API ───────────────────────────────────────────────────────────

    def load_all(self) -> List[DocumentPage]:
        """Load every PDF in the source directory."""
        if self.source_dir:
            return self._load_from_directory()
        elif self.azure_container:
            return self._load_from_azure()
        else:
            raise ValueError("Provide either source_dir or azure_container.")

    def load_file(self, pdf_path: str) -> List[DocumentPage]:
        """Load a single PDF file and return its pages."""
        return self._extract_pages(Path(pdf_path))

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _load_from_directory(self) -> List[DocumentPage]:
        pdf_files = sorted(self.source_dir.glob("**/*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDFs found in {self.source_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDFs in {self.source_dir}")
        pages = []
        for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
            pages.extend(self._extract_pages(pdf_path))
        logger.info(f"Extracted {len(pages)} pages total")
        return pages

    def _load_from_azure(self) -> List[DocumentPage]:
        """Download blobs from Azure and extract pages."""
        import tempfile
        from azure.storage.blob import BlobServiceClient

        conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        client = BlobServiceClient.from_connection_string(conn_str)
        container = client.get_container_client(self.azure_container)

        pages = []
        with tempfile.TemporaryDirectory() as tmp:
            for blob in tqdm(list(container.list_blobs()), desc="Downloading from Azure"):
                if not blob.name.endswith(".pdf"):
                    continue
                local_path = Path(tmp) / Path(blob.name).name
                with open(local_path, "wb") as f:
                    f.write(container.download_blob(blob.name).readall())
                pages.extend(self._extract_pages(local_path))
        return pages

    def _extract_pages(self, pdf_path: Path) -> List[DocumentPage]:
        """Extract text from every page of a single PDF using PyMuPDF."""
        pages = []
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()

                # skip near-blank pages (headers/footers only)
                if len(text) < self.min_page_chars:
                    continue

                pages.append(DocumentPage(
                    source=pdf_path.name,
                    page_number=page_num + 1,
                    text=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num + 1,
                        "total_pages": len(doc),
                    }
                ))
            doc.close()
        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")
        return pages
