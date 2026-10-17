"""Split pages into citation-sized chunks without crossing a page boundary.

Page-bounded chunks keep a citation honest: the page number on a chunk is the
page the text was extracted from, not an estimate after a sliding window.
"""

from __future__ import annotations

import re

from fin_doc_analyst.schemas import Chunk

_HEADING = re.compile(r"^section:\s*(.+)$", re.I)


def is_heading(line: str) -> str | None:
    match = _HEADING.match(line.strip())
    if not match:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title[:80] or None


def split_text(text: str, max_chars: int = 700, overlap: int = 80) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            breakpoint = text.rfind(" ", start, end)
            if breakpoint > start + max_chars // 2:
                end = breakpoint
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_pages(
    doc_id: str,
    doc_name: str,
    pages: list[tuple[int, str]],
    max_chars: int = 700,
    overlap: int = 80,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for page_number, text in pages:
        section = "Body"
        blocks: list[tuple[str, list[str]]] = []
        current: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            heading = is_heading(line)
            if heading:
                if current:
                    blocks.append((section, current))
                    current = []
                section = heading
                continue
            current.append(line)
        if current:
            blocks.append((section, current))
        if not blocks and text.strip():
            blocks = [("Body", [text.strip()])]
        for block_section, lines in blocks:
            for piece in split_text("\n".join(lines), max_chars=max_chars, overlap=overlap):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}:{page_number}:{index}",
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page=page_number,
                        section=block_section,
                        text=piece,
                    )
                )
                index += 1
    return chunks
