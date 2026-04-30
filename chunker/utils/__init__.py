#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) utilities package.

This package provides document processing and chunking functionality for RAG systems,
enabling intelligent extraction of structured information from PDF documents.

Modules:
    core: Main DocumentProcessor class that orchestrates the processing pipeline.
    extractor: PDF text extraction with font and positioning information.
    analyzer: Font analysis for header detection and document structure.
    chunker: Intelligent document chunking with hierarchy preservation.
    models: Data classes and enums for document processing.
    query_engine: Advanced chunk querying and search functionality.

Example:
    >>> from lib import DocumentProcessor
    >>> processor = DocumentProcessor()
    >>> result = processor.process_document("document.pdf")
    >>> chunks = result['chunks']
"""

from .models import (
    ChunkMetadata,
    DocumentChunk,
    FontAnalysis,
    FontInfo,
    HeaderInfo,
    HierarchyStructure,
    ProcessingStats,
    ProcessingResult,
    TextElement,
)
from .analyzer import FontAnalyzer
from .chunker import DocumentChunker
from .core import DocumentProcessor
from .extractor import DocumentExtractor

__all__ = [
    # Main API
    "DocumentProcessor",
    # Core components
    "DocumentExtractor",
    "FontAnalyzer",
    "DocumentChunker",
    # Data models
    "FontInfo",
    "TextElement",
    "HeaderInfo",
    "ChunkMetadata",
    "DocumentChunk",
    "ProcessingStats",
    "ProcessingResult",
    "FontAnalysis",
    "HierarchyStructure",
]

# Version information
__version__ = "1.0.0"
__author__ = "RAG Processing Team"
