"""
Shared state, initialization, and helper functions for document tools.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import get_config
from ..logging_config import get_logger

logger = get_logger(__name__)

# Global state (set during initialization)
_chunks_dir: Optional[Path] = None


def initialize_tools(chunks_dir: str) -> None:
    """
    Initialize tools with chunks directory.
    """
    global _chunks_dir
    _chunks_dir = Path(chunks_dir)


def get_chunks_dir() -> Optional[Path]:
    """Return the current chunks directory path."""
    return _chunks_dir


def _get_available_document_names() -> List[str]:
    """Return list of document folder names (same logic as discover_documents). Used for error messages."""
    global _chunks_dir
    if not _chunks_dir or not _chunks_dir.exists():
        return []
    documents = []
    for outline_path in _chunks_dir.rglob("outline.txt"):
        doc_dir = outline_path.parent
        try:
            rel = doc_dir.relative_to(_chunks_dir)
            documents.append(str(rel))
        except ValueError:
            continue
    for chunk_dir in _chunks_dir.rglob("chunk"):
        if chunk_dir.is_dir():
            doc_dir = chunk_dir.parent
            try:
                rel = doc_dir.relative_to(_chunks_dir)
                name = str(rel)
                if name not in documents:
                    documents.append(name)
            except ValueError:
                continue
    return documents


def _load_document_chunks(document_name: str) -> Dict[str, Dict[str, Any]]:
    """Load chunks from document_name/chunk/*.json (one JSON per chunk)."""
    global _chunks_dir

    if not _chunks_dir:
        return {}

    document_path = _chunks_dir / document_name
    chunk_dir = document_path / "chunk"
    if not chunk_dir.is_dir():
        logger.warning("Chunk folder not found", extra={"extra_fields": {"path": str(chunk_dir)}})
        return {}

    chunks: Dict[str, Dict[str, Any]] = {}
    for file_path in sorted(chunk_dir.glob("*.json")):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunk_id = data.get("chunk_id") or file_path.stem
            chunks[chunk_id] = data
        except Exception as e:
            logger.warning("Failed to load chunk file", extra={"extra_fields": {"path": str(file_path), "error": str(e)}})

    if chunks:
        logger.info("Loaded document chunks", extra={"extra_fields": {"document_name": document_name, "count": len(chunks)}})
    return chunks


def get_planner_tools() -> List[Any]:
    """
    Returns tools for the Planner node (Node 1).
    
    These tools are used for document discovery and structure exploration:
        - discover_documents: Find available documents
        - read_outline: Get human-readable TOC
    """
    from .discover_documents import discover_documents
    from .read_outline import read_outline
    return [discover_documents]#, read_outline]


def get_retriever_tools() -> List[Any]:
    """
    Returns tools for the Retriever node (Node 2).
    
    These tools are used for chunk retrieval:
        - get_chunks_by_section: Retrieve chunks by exact section number
        - search_chunks: Keyword search across chunks
        - regex_search_chunks: Regex search across chunks; returns section numbers and headings
    """
    from .read_outline import read_outline
    from .get_chunks_by_section import get_chunks_by_section
    from .search_chunks import search_chunks
    from .regex_search_chunks import regex_search_chunks
    return [read_outline, get_chunks_by_section, search_chunks, regex_search_chunks]


def get_all_tools() -> List[Any]:
    """
    Returns all document tools for LangChain agent.
    """
    from .discover_documents import discover_documents
    from .read_document_index import read_document_index
    from .read_outline import read_outline
    from .get_chunks_by_section import get_chunks_by_section
    from .search_chunks import search_chunks
    from .regex_search_chunks import regex_search_chunks
    return [discover_documents, read_document_index, read_outline, get_chunks_by_section, search_chunks, regex_search_chunks]
