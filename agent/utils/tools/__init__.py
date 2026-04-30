"""
Tools for document index and chunk retrieval.

The tools are organized by node responsibility in the 3-node pipeline:
    - Planner tools: discover_documents (via get_planner_tools)
    - Retriever tools: read_outline, get_chunks_by_section, search_chunks, regex_search_chunks
      (retriever builds contextual tools internally with document path injected)
"""

from .discover_documents import discover_documents
from .read_document_index import read_document_index
from .read_outline import read_outline
from .get_chunks_by_section import get_chunks_by_section
from .search_chunks import search_chunks
from .regex_search_chunks import regex_search_chunks
from ._common import (
    initialize_tools,
    get_planner_tools,
    get_retriever_tools,
)

__all__ = [
    # Individual tools
    'discover_documents',
    'read_document_index',
    'read_outline',
    'get_chunks_by_section',
    'search_chunks',
    'regex_search_chunks',
    # Tool collections (get_all_tools available via _common for future use)
    'get_planner_tools',
    'get_retriever_tools',
    # Initialization
    'initialize_tools'
]
