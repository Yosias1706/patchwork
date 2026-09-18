"""Deterministic, overlap-aware chunking that keeps source offsets."""

from __future__ import annotations

import hashlib

from .models import TextChunk


class TextSplitter:
    """Split text near natural boundaries while retaining a configurable overlap."""

    _BOUNDARIES = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ")

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document_id: str, text: str) -> list[TextChunk]:
        text = text.strip()
        if not text:
            return []

        chunks: list[TextChunk] = []
        start = 0
        ordinal = 0
        length = len(text)
        while start < length:
            hard_end = min(start + self.chunk_size, length)
            end = self._natural_end(text, start, hard_end)
            left = start
            right = end
            while left < right and text[left].isspace():
                left += 1
            while right > left and text[right - 1].isspace():
                right -= 1
            if left < right:
                chunk_text = text[left:right]
                chunk_identity = f"{document_id}:{ordinal}:{chunk_text}"
                digest = hashlib.sha256(chunk_identity.encode()).hexdigest()
                chunks.append(
                    TextChunk(
                        id=digest,
                        document_id=document_id,
                        ordinal=ordinal,
                        text=chunk_text,
                        start_offset=left,
                        end_offset=right,
                    )
                )
                ordinal += 1
            if end >= length:
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def _natural_end(self, text: str, start: int, hard_end: int) -> int:
        if hard_end == len(text):
            return hard_end
        # Prefer a boundary in the latter 40% of the proposed window. This avoids
        # tiny chunks created by an early newline or sentence break.
        floor = start + int(self.chunk_size * 0.60)
        window = text[start:hard_end]
        for boundary in self._BOUNDARIES:
            position = window.rfind(boundary)
            if position >= floor - start:
                return start + position + len(boundary)
        return hard_end
