"""
Tool for reading the document_index.json file for a document.
"""

import json
import time
from typing import Any, Dict

from langchain_core.tools import tool

from ..errors import DocumentNotFoundError
from ..logging_config import get_logger
from ._common import get_chunks_dir, _get_available_document_names

logger = get_logger(__name__)


@tool
def read_document_index(document_name: str) -> Dict[str, Any]:
    """
    Reads the document_index.json file for a document.
    
    Purpose:
        Retrieves the structured document index containing hierarchical section information, metadata,
        and programmatic access to document structure. This provides a comprehensive overview of the
        document's organization and enables efficient navigation to specific sections.
    
    When to use:
        - To understand the document structure and hierarchy before retrieving chunks
        - To find specific section numbers when the user mentions sections (e.g., "section 1.2")
        - To get an overview of all sections and their organization
        - To identify section numbers that match a query before using get_chunks_by_section()
        - When you need programmatic access to document structure (not just human-readable format)
    
    Args:
        document_name (str): Document folder name (e.g., "CrowdStrike", "CiscoDuo")
            This should match one of the document names returned by discover_documents().
            Example: "CrowdStrike"
    
    Returns:
        Dict[str, Any]: Document index containing:
            - "document_name": Name of the document
            - "generated_at": ISO timestamp of when the index was generated
            - "outline_text": Human-readable outline text
            - "structured_index": Dictionary with:
                - "sections": Hierarchical tree of sections with metadata (section numbers, titles, chunk_ids, page ranges, word counts)
                - "total_sections": Total number of sections in the document
                - "max_depth": Maximum nesting depth of the hierarchy
        Returns an empty dict {} if the document doesn't exist, index file is missing, or read fails.
    
    Usage examples:
        - User query: "What sections are in the CrowdStrike document?" → Call read_document_index("CrowdStrike") to see all sections
        - User query: "Find information about section 2.3" → Call read_document_index() first to verify section 2.3 exists, then use get_chunks_by_section()
        - User query: "Show me the structure of the ZScalar document" → Call read_document_index("ZScalar") to get hierarchical structure
    
    Related tools:
        - Use discover_documents() first if you don't know the document name
        - Use read_outline() if you prefer human-readable text format over structured data
        - After finding section numbers, use get_chunks_by_section() to retrieve chunks for specific sections
        - Use search_chunks() if you need to search by keywords instead of section numbers
    """
    logger.info("read_document_index called", extra={"extra_fields": {"document_name": document_name}})
    _chunks_dir = get_chunks_dir()
    if not _chunks_dir:
        return {}
    
    index_path = _chunks_dir / document_name / "document_index.json"
    if not index_path.exists():
        available = _get_available_document_names()
        logger.warning("Document index not found", extra={"extra_fields": {"document_name": document_name, "available": available}})
        raise DocumentNotFoundError(
            f"Document not found: '{document_name}'. Use an EXACT path from discover_documents().",
            available_documents=available,
        )

    try:
        start_time = time.time()
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        elapsed = time.time() - start_time
        logger.info("read_document_index loaded", extra={"extra_fields": {"document_name": document_name, "elapsed": round(elapsed, 2)}})
        return data
    except DocumentNotFoundError:
        raise
    except Exception as e:
        logger.exception("Failed to load document index")
        raise
