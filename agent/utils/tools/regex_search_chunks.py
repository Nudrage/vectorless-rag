"""
Tool for regex-based search across all chunks in a document.
"""

import re
import time
from typing import Any, Dict, List

from langchain_core.tools import tool

from ..logging_config import get_logger
from ._common import _load_document_chunks

logger = get_logger(__name__)


@tool
def regex_search_chunks(document_name: str, pattern: str) -> List[Dict[str, Any]]:
    """
    Performs a regex search across all chunks in a document and returns section numbers and headings of matching chunks.
    
    Purpose:
        Finds chunks whose content or header matches a regular expression. Returns only section number and
        heading for each match (not full content), useful for discovering which sections mention a pattern
        (e.g. API names, error codes, specific formats).
    
    When to use:
        - When you need pattern-based search (e.g. "all sections mentioning IDs like ABC-123", "endpoints containing /v2/")
        - To find sections that match a specific format or convention (regex)
        - When keyword search is not enough and you need alternation, anchors, or character classes
    
    Args:
        document_name (str): Document folder name (e.g., "CrowdStrike", "Cisco Duo")
            This should match one of the document names returned by discover_documents().
        
        pattern (str): Regular expression pattern to search for. Applied to each chunk's header and content.
            Uses Python regex syntax. Search is applied to the raw text of each chunk.
            Examples: r"\\bAPI\\b", "error_[A-Z0-9]+", "/api/v\\d+/"
    
    Returns:
        List[Dict[str, Any]]: List of matches, each with:
            - "section_number": Section number of the chunk (e.g. "1.2.3")
            - "header": Heading/title of the chunk
            - "chunk_id": Chunk identifier (for reference or get_chunks_by_section)
        Returns an empty list if no chunks match or the document does not exist.
        Raises ValueError if the regex pattern is invalid.
    
    Usage examples:
        - Find sections mentioning "API": regex_search_chunks("DocumentName", r"\\bAPI\\b")
        - Find error codes: regex_search_chunks("DocumentName", "error_[0-9]+")
        - Find versioned paths: regex_search_chunks("DocumentName", "/v[12]/")
    
    Related tools:
        - Use get_chunks_by_section() with a returned section_number to fetch full chunk content
        - Use search_chunks() for plain keyword search without regex
    """
    logger.info("regex_search_chunks called", extra={"extra_fields": {"document_name": document_name, "pattern_length": len(pattern)}})
    start_time = time.time()

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")

    all_chunks = _load_document_chunks(document_name)

    if not all_chunks:
        elapsed = time.time() - start_time
        logger.info("regex_search_chunks done", extra={"extra_fields": {"result_count": 0, "elapsed": round(elapsed, 2)}})
        return []

    results = []
    for chunk_id, chunk_data in all_chunks.items():
        header = chunk_data.get("header") or ""
        content = chunk_data.get("content") or ""
        if compiled.search(header) or compiled.search(content):
            results.append({
                "section_number": chunk_data.get("section_number") or "",
                "header": header,
                "chunk_id": chunk_id,
            })

    elapsed = time.time() - start_time
    logger.info("regex_search_chunks done", extra={"extra_fields": {"result_count": len(results), "elapsed": round(elapsed, 2)}})
    return results
