"""Extract text per page. Page numbers are 1-based, matching the citation UI."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class PdfParseError(ValueError):
    """Raised when bytes are not a readable PDF."""


def extract_pages(source: Path | bytes) -> list[tuple[int, str]]:
    try:
        if isinstance(source, bytes):
            from io import BytesIO

            reader = PdfReader(BytesIO(source))
        else:
            reader = PdfReader(str(source))
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append((index, page.extract_text() or ""))
        return pages
    except Exception as exc:  # noqa: BLE001 — pypdf raises a wide mix of errors
        raise PdfParseError(f"Could not read PDF: {exc}") from exc
