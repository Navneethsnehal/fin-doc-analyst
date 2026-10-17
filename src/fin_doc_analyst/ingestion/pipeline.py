"""Ingest a PDF into chunks with document, page, and section metadata."""

from __future__ import annotations

import re
from pathlib import Path

from fin_doc_analyst.ingestion.chunker import chunk_pages
from fin_doc_analyst.ingestion.pdf import extract_pages
from fin_doc_analyst.schemas import Chunk, DocumentInfo


def safe_doc_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-")
    return cleaned or "document"


def ingest_pdf(
    filename: str, data: bytes, doc_id: str | None = None
) -> tuple[DocumentInfo, list[Chunk]]:
    pages = extract_pages(data)
    resolved_id = doc_id or safe_doc_id(filename)
    chunks = chunk_pages(resolved_id, Path(filename).name, pages)
    info = DocumentInfo(
        doc_id=resolved_id,
        doc_name=Path(filename).name,
        pages=len(pages),
        chunks=len(chunks),
        synthetic=True,
    )
    return info, chunks
