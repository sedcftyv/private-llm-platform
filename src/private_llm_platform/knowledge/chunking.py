"""Stable Markdown and text document chunking."""

import hashlib
import re
from pathlib import Path

from .models import DocumentChunk

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_DOCUMENT_ID_PATTERN = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


def chunk_document(
    path: Path,
    *,
    document_id: str | None = None,
    document_title: str | None = None,
    version: str = "unknown",
    max_chars: int = 500,
    overlap_chars: int = 80,
) -> list[DocumentChunk]:
    """Load and split a Markdown or text document.

    Args:
        path: Source `.md` or `.txt` file.
        document_id: Stable document identifier, derived from the filename by default.
        document_title: Display title, derived from the first Markdown H1 by default.
        version: Source document version stored with every chunk.
        max_chars: Maximum characters in a chunk.
        overlap_chars: Maximum trailing characters reused by the next chunk.

    Returns:
        Stable, non-empty document chunks.

    Raises:
        ValueError: If the file type or chunking limits are invalid.
    """

    if path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError("only Markdown and text documents are supported")
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between zero and max_chars")

    text = _normalize_text(path.read_text(encoding="utf-8"))
    if not text:
        return []

    resolved_document_id = document_id or _document_id(path.stem)
    sections, markdown_title = _sections(text, is_markdown=path.suffix.lower() == ".md")
    resolved_title = document_title or markdown_title or path.stem

    chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()
    for section, content in sections:
        for fragment in _split_content(content, max_chars, overlap_chars):
            chunk_id = _chunk_id(resolved_document_id, section, fragment)
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=resolved_document_id,
                    document_title=resolved_title,
                    section=section,
                    content=fragment,
                    source_path=path.as_posix(),
                    version=version,
                )
            )
    return chunks


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    normalized = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def _document_id(stem: str) -> str:
    value = _DOCUMENT_ID_PATTERN.sub("-", stem).strip("-").lower()
    if not value:
        raise ValueError("could not derive a document_id from the filename")
    return value


def _sections(text: str, *, is_markdown: bool) -> tuple[list[tuple[str, str]], str]:
    if not is_markdown:
        return [("正文", text)], ""

    title = ""
    current_section = "正文"
    current_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_section, content))
        current_lines.clear()

    for line in text.splitlines():
        match = _HEADING_PATTERN.match(line)
        if not match:
            current_lines.append(line)
            continue

        flush()
        level = len(match.group(1))
        heading = match.group(2).strip()
        if level == 1 and not title:
            title = heading
        current_section = heading

    flush()
    return sections, title


def _split_content(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(paragraph, max_chars, overlap_chars))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        chunks.append(current)
        overlap = current[-overlap_chars:] if overlap_chars else ""
        candidate = f"{overlap}\n\n{paragraph}" if overlap else paragraph
        current = candidate if len(candidate) <= max_chars else paragraph

    if current:
        chunks.append(current)
    return chunks


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap_chars
    return chunks


def _chunk_id(document_id: str, section: str, content: str) -> str:
    value = f"{document_id}\0{section}\0{content}".encode()
    digest = hashlib.sha256(value).hexdigest()[:12]
    return f"{document_id}-{digest}"
