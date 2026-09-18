"""Persistent, local-first RAG document search."""

from .config import Settings
from .service import RAGService

__all__ = ["RAGService", "Settings"]
