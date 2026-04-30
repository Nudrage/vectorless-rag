"""
Tool for keyword-based search across all chunks in a document.
"""

import time
from typing import Any, Dict, List

from langchain_core.tools import tool

from ..logging_config import get_logger
from ._common import _load_document_chunks

logger = get_logger(__name__)


@tool
def search_chunks(document_name: str, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Performs keyword-based search across all chunks in a document.
    
    Purpose:
        Searches through all document chunks using keyword matching to find relevant content.
        This tool uses a relevance scoring system that prioritizes matches in headers and exact
        phrase matches. Results are automatically sorted by relevance score (highest first).
        This is the primary tool for finding information when section numbers are unknown.
    
    When to use:
        - For keyword-based searches when you don't know the specific section number
        - When the user query contains specific terms or topics to search for
        - For broad topic queries that might span multiple sections
        - When you need to find all mentions of a specific concept, term, or phrase
        - As an alternative to get_chunks_by_section() when section numbers are unknown
        - To search across the entire document for relevant information
    
    Args:
        document_name (str): Document folder name (e.g., "CrowdStrike", "CiscoDuo")
            This should match one of the document names returned by discover_documents().
            Example: "ZScalar"
        
        query (str): Search query for keyword matching. Can be:
            - Single words: "authentication"
            - Multiple words: "firewall logs"
            - Phrases: "API endpoint configuration"
            - Specific terms from the user's query
            The search is case-insensitive and matches words in both headers and content.
            Example: "event types" or "log schema"
        
        max_results (int, optional): Maximum number of chunks to return. Defaults to 10.
            Use a higher value (e.g., 20-50) if you need more comprehensive results.
            Use a lower value (e.g., 5) if you want only the most relevant chunks.
            Example: 15
    
    Returns:
        List[Dict[str, Any]]: List of relevant chunks sorted by relevance score (highest first).
        Each chunk dictionary contains:
            - "chunk_id": Unique identifier for the chunk
            - "header": Section header/title text
            - "content": Full text content of the chunk
            - "section_number": The section number this chunk belongs to
            - "relevance_score": Numeric score indicating relevance (higher = more relevant)
            - "metadata": Additional metadata including page ranges, font info, etc.
        
        Scoring system:
            - Header matches: +2.0 points per matching word
            - Content matches: +1.0 points per matching word
            - Exact phrase match in header: +5.0 bonus
            - Exact phrase match in content: +3.0 bonus
            - Results are sorted by total score (descending)
        
        Returns an empty list [] if no matches are found, or if the document doesn't exist.
    
    Usage examples:
        - User query: "Find information about authentication" → Call search_chunks("DocumentName", "authentication")
        - User query: "What are the firewall log fields?" → Call search_chunks("DocumentName", "firewall log fields")
        - User query: "Tell me about API configuration" → Call search_chunks("DocumentName", "API configuration", max_results=15)
        - After discovering document: Use search_chunks() with keywords extracted from the user's query
        - Broad search: If user asks about "events", call search_chunks("DocumentName", "events", max_results=20)
    
    Best practices:
        - Extract key terms from the user's query to use as search terms
        - Use specific phrases rather than generic words for better results
        - If initial search returns too few results, try broader terms or increase max_results
        - If initial search returns too many results, refine the query with more specific terms
        - Combine multiple searches if the query covers multiple topics
        - Review relevance scores to understand result quality
    
    Related tools:
        - Use discover_documents() first if you don't know the document name
        - Use read_document_index() or read_outline() to understand structure before searching
        - Use get_chunks_by_section() if you know the exact section number (more precise than search)
        - Can be combined with get_chunks_by_section() to search within specific sections
    """
    logger.info("search_chunks called", extra={"extra_fields": {"document_name": document_name, "query": query[:50], "max_results": max_results}})
    start_time = time.time()

    all_chunks = _load_document_chunks(document_name)

    logger.debug("Searching chunks", extra={"extra_fields": {"count": len(all_chunks)}})
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_chunks = []
    
    for chunk_id, chunk_data in all_chunks.items():
        header = chunk_data.get('header', '').lower()
        content = chunk_data.get('content', '').lower()
        
        score = 0.0
        
        # Header matches are more important
        for word in query_words:
            if word in header:
                score += 2.0
            if word in content:
                score += 1.0
        
        # Exact phrase match bonus
        if query_lower in header:
            score += 5.0
        if query_lower in content:
            score += 3.0
        
        if score > 0:
            chunk_with_score = chunk_data.copy()
            chunk_with_score['relevance_score'] = score
            scored_chunks.append(chunk_with_score)
    
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    result = scored_chunks[:max_results]
    elapsed = time.time() - start_time
    logger.info("search_chunks done", extra={"extra_fields": {"result_count": len(result), "matches": len(scored_chunks), "elapsed": round(elapsed, 2)}})
    return result
