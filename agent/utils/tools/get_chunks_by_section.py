"""
Tool for retrieving chunks that belong to a specific section number.
"""

import time
from typing import Any, Dict, List

from langchain_core.tools import tool

from ..logging_config import get_logger
from ._common import _load_document_chunks

logger = get_logger(__name__)


@tool
def get_chunks_by_section(document_name: str, section_number: str) -> List[Dict[str, Any]]:
    """
    Retrieves chunks that belong to a specific section number (exact match only, no subsections).
    
    Purpose:
        Retrieves all document chunks that belong to an exact section number. This is the most
        precise way to retrieve information when you know the specific section number. The tool
        performs exact matching only - it will NOT return chunks from subsections.
    
    When to use:
        - When the user query explicitly mentions a specific section number (e.g., "section 1.2", "chapter 3.4")
        - When you have identified a relevant section number from read_document_index() or read_outline()
        - For precise retrieval of content from a known section
        - When you need all chunks from a specific section (not just keyword matches)
        - After exploring document structure and identifying the exact section number needed
    
    Args:
        document_name (str): Document folder name (e.g., "CrowdStrike", "CiscoDuo")
            This should match one of the document names returned by discover_documents().
            Example: "ZScalar"
        
        section_number (str): Section number to match exactly. Must be the complete section number.
            Examples:
                - "1" matches only section 1 (not 1.1, 1.2, etc.)
                - "1.2" matches only section 1.2 (not 1.2.1, 1.2.2, etc.)
                - "2.3.4" matches only section 2.3.4 (not 2.3.4.1, etc.)
            IMPORTANT: This is an exact match - subsections are NOT included.
            To find section numbers, use read_document_index() or read_outline() first.
    
    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries belonging to the exact section number.
        Each chunk dictionary contains:
            - "chunk_id": Unique identifier for the chunk (e.g., "introduction_001")
            - "header": Section header/title text
            - "content": Full text content of the chunk
            - "section_number": The section number this chunk belongs to
            - "metadata": Additional metadata including page ranges, font info, hierarchy info, etc.
        Returns an empty list [] if no chunks match the exact section number, or if the document doesn't exist.
    
    Usage examples:
        - User query: "Get me information from section 1.2 of CrowdStrike" → Call get_chunks_by_section("CrowdStrike", "1.2")
        - User query: "What's in chapter 3?" → First use read_outline() to find section numbers, then call get_chunks_by_section() with the exact section number
        - After reading index: If read_document_index() shows section "2.3" is relevant, call get_chunks_by_section("DocumentName", "2.3")
        - User query: "Show me section 4.5.6" → Call get_chunks_by_section("DocumentName", "4.5.6")
    
    Important notes:
        - This tool performs EXACT matching only. Section "1.2" will NOT return chunks from "1.2.1" or "1.2.2"
        - If you need subsections, you must call this tool separately for each subsection number
        - If the section number doesn't exist, an empty list is returned
        - Use read_document_index() or read_outline() to verify section numbers exist before calling this tool
    
    Related tools:
        - Use discover_documents() first if you don't know the document name
        - Use read_document_index() or read_outline() to find section numbers before calling this tool
        - Use search_chunks() if you don't know the section number and need keyword-based search instead
        - Combine results from multiple section numbers if you need information from multiple sections
    """
    logger.info("get_chunks_by_section called", extra={"extra_fields": {"document_name": document_name, "section": section_number}})
    start_time = time.time()

    all_chunks = _load_document_chunks(document_name)

    section_chunks = []
    for chunk_id, chunk_data in all_chunks.items():
        chunk_section = chunk_data.get('section_number')
        if chunk_section and chunk_section == section_number:
            section_chunks.append(chunk_data)

    elapsed = time.time() - start_time
    logger.info("get_chunks_by_section done", extra={"extra_fields": {"count": len(section_chunks), "elapsed": round(elapsed, 2)}})
    return section_chunks
