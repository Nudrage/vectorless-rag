#!/usr/bin/env python3
"""
Data models for document processing and chunking.

This module defines all data classes and enums used throughout the RAG
document processing pipeline, including font information, text elements,
headers, chunks, and processing statistics.

Example:
    >>> from lib.models import DocumentChunk, ChunkMetadata
    >>> metadata = ChunkMetadata(
    ...     header_text="Introduction",
    ...     header_level=1,
    ...     font_size=14.0,
    ...     font_name="Arial",
    ...     page=1,
    ...     start_page=1,
    ...     end_page=2
    ... )
    >>> chunk = DocumentChunk(
    ...     chunk_id="intro_001",
    ...     header="Introduction",
    ...     content="Document content here...",
    ...     metadata=metadata
    ... )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict

class _ProcessingResultRequired(TypedDict):
    """Result of document processing. On success, 'error' is absent."""

    success: bool
    document_info: Optional[Dict[str, Any]]
    chunks: None
    chunks_location: Optional[str]
    hierarchy: Dict[str, Any]
    font_analysis: Dict[str, Any]
    stats: Optional[Dict[str, Any]]
    total_chunks: int


class ProcessingResult(_ProcessingResultRequired, total=False):
    error: str


@dataclass
class FontInfo:
    """
    [PURPOSE]
    Data model representing font properties extracted from PDF text elements. This dataclass
    stores essential font information (size, name, weight, style) that is used throughout the
    document processing pipeline for header detection, content classification, and font-based
    analysis. Font size is particularly important as it's the primary signal for identifying
    headers versus body text.
    
    [FIELD DESCRIPTIONS]
    - size: Float representing the font size in points (e.g., 12.0, 14.5, 16.0). This is the
      most critical property for header detection - larger fonts typically indicate headers.
      The size is extracted from PDF span data and may be rounded to an integer using round()
      during font analysis for consistency. Required field.
    - name: String containing the font family name (e.g., "Arial", "Times New Roman", "Calibri").
      Extracted directly from PDF font metadata. Used for font analysis and document structure
      understanding. Required field.
    - weight: String specifying font weight ("normal", "bold", "light", etc.). Defaults to "normal".
      Used as a secondary signal for header detection (bold text is more likely to be headers).
      Optional field with default value.
    - style: String specifying font style ("normal", "italic", "oblique", etc.). Defaults to "normal".
      Used for text classification and formatting preservation. Optional field with default value.
    
    [DATA FLOW]
    When instantiated:
    1. size and name must be provided (required fields).
    2. weight defaults to "normal" if not provided.
    3. style defaults to "normal" if not provided.
    4. Python's dataclass mechanism automatically generates __init__, __repr__, __eq__ methods.
    
    [USAGE]
    Typically created by DocumentExtractor when extracting text elements from PDFs. The FontInfo
    objects are attached to TextElement objects and used by FontAnalyzer to identify headers
    based on size distribution patterns.
    
    [SIDE EFFECTS]
    - No side effects. This is a pure data container.
    - No validation or transformation occurs during instantiation (dataclass default behavior).
    """
    size: float  # Font size in points (primary signal for header detection)
    name: str  # Font family name (e.g., "Arial", "Times New Roman")
    weight: str = "normal"  # Font weight ("normal", "bold", etc.)
    style: str = "normal"  # Font style ("normal", "italic", etc.)


@dataclass
class TextElement:
    """
    [PURPOSE]
    Data model representing a single text element extracted from a PDF page. This dataclass
    encapsulates a line of text along with its font properties and positioning information.
    TextElement objects are the fundamental building blocks of the document processing pipeline -
    they are extracted from PDFs, analyzed for font patterns, and grouped into chunks. The
    positioning information (bbox, page, line_number) is crucial for maintaining document
    order and spatial relationships.
    
    [FIELD DESCRIPTIONS]
    - text: String containing the actual text content of the element (typically one line from
      the PDF). This is the raw text extracted from PDF spans, joined together for the line.
      May contain whitespace that should be stripped before processing. Required field.
    - font: FontInfo object containing font properties (size, name, weight, style) for this
      text element. The font information is used by FontAnalyzer to identify headers and by
      DocumentChunker to group content. Required field.
    - page: Integer representing the page number where this element appears (1-indexed, so page 1
      is the first page). Used for maintaining document order and tracking content location.
      Required field.
    - bbox: Tuple of four floats representing the bounding box coordinates (x0, y0, x1, y1) in
      PDF coordinate space. The bbox defines the rectangular area occupied by the text on the
      page. Used for spatial analysis, header merging (checking vertical proximity), and
      position validation. Required field.
    - line_number: Integer representing the line number on the page (0-indexed or 1-indexed
      depending on extraction logic). Used for maintaining order within a page and debugging.
      Required field.
    
    [DATA FLOW]
    When instantiated:
    1. All fields must be provided during instantiation (no defaults).
    2. Python's dataclass mechanism automatically generates __init__, __repr__, __eq__ methods.
    3. The text is typically stripped of leading/trailing whitespace before creating the element.
    
    [USAGE]
    Created by DocumentExtractor.extract_text_elements() when processing PDF pages. TextElement
    objects are collected into a list and passed to FontAnalyzer for font pattern analysis,
    then to DocumentChunker for chunk creation.
    
    [SIDE EFFECTS]
    - No side effects. This is a pure data container.
    - No validation or transformation occurs during instantiation (dataclass default behavior).
    """
    text: str  # Actual text content (typically one line from PDF)
    font: FontInfo  # Font properties (size, name, weight, style)
    page: int  # Page number (1-indexed)
    bbox: Tuple[float, float, float, float]  # Bounding box (x0, y0, x1, y1)
    line_number: int  # Line number on the page


@dataclass
class HeaderInfo:
    """
    [PURPOSE]
    Data model representing a detected header in the document with hierarchical relationships.
    This dataclass stores header information including text, hierarchy level, font properties,
    and parent-child relationships. HeaderInfo objects form a tree structure representing the
    document's organization. The parent and children references enable efficient hierarchy
    traversal and chunk creation. Headers can be merged from multiple consecutive text elements
    if they appear to be parts of a single wrapped header line.
    
    [FIELD DESCRIPTIONS]
    - text: String containing the header text content. This may be a single line or merged
      from multiple consecutive text elements if the header wraps across lines. The text is
      used for chunk headers and hierarchy building. Required field.
    - level: Integer representing the hierarchy level (1 = top level/root, 2 = subsection,
      3 = sub-subsection, etc.). The level is determined by font size relationships and
      document structure. Lower numbers indicate higher-level sections. Required field.
    - font: FontInfo object containing font properties for the header. The font size is
      particularly important as it's used to determine hierarchy relationships. Required field.
    - page: Integer representing the page number where the header appears (1-indexed). Used for
      maintaining document order and tracking header locations. Required field.
    - position: Integer representing the position index of the header in the document's text
      elements list. This is the index in the TextElement list, used for ordering and content
      extraction. Required field.
    - parent: Optional reference to the parent HeaderInfo object. If this header is a subsection,
      parent points to the higher-level header. If this is a root header, parent is None.
      Used for building hierarchical relationships and section numbering. Optional field.
    - children: List of HeaderInfo objects representing child headers (subsections) under this
      header. The list is maintained in document order. Used for hierarchy traversal and
      chunk organization. Defaults to empty list if not provided.
    - is_merged: Boolean indicating whether this header was merged from multiple consecutive
      text elements. If True, the header text combines multiple lines that were detected as
      parts of a single header. Used for tracking header merging operations. Defaults to False.
    - original_positions: List of integers containing the original position indices of text
      elements that were merged to create this header. Only populated if is_merged is True.
      Used for debugging and tracking merged header sources. Defaults to empty list.
    - end_position: Optional integer representing the end position of merged headers. For
      merged headers spanning multiple elements, this indicates the last element's position.
      Used for tracking the full span of merged headers. Optional field.
    
    [DATA FLOW]
    When instantiated:
    1. Required fields (text, level, font, page, position) must be provided.
    2. Optional fields (parent, children, is_merged, original_positions, end_position) use
       defaults if not provided.
    3. If children is None, it's initialized as empty list via field(default_factory=list).
    4. Python's dataclass mechanism automatically generates __init__, __repr__, __eq__ methods.
    5. Parent and children references create a tree structure (circular references are allowed).
    
    [USAGE]
    Created by DocumentChunker._identify_headers() when detecting headers from text elements.
    HeaderInfo objects are then processed by _build_font_size_hierarchy() to establish parent-child
    relationships, and used by _create_chunks_from_headers() to create document chunks.
    
    [SIDE EFFECTS]
    - No side effects. This is a pure data container.
    - Parent and children references create object graphs (tree structures).
    - No validation or transformation occurs during instantiation (dataclass default behavior).
    """
    text: str  # Header text content (may be merged from multiple lines)
    level: int  # Hierarchy level (1 = root, 2 = subsection, etc.)
    font: FontInfo  # Font properties for the header
    page: int  # Page number where header appears (1-indexed)
    position: int  # Position index in document's text elements list
    parent: Optional['HeaderInfo'] = None  # Reference to parent header (None if root)
    children: List['HeaderInfo'] = field(default_factory=list)  # List of child headers
    is_merged: bool = False  # Whether header was merged from multiple elements
    original_positions: List[int] = field(default_factory=list)  # Original positions of merged elements
    end_position: Optional[int] = None  # End position for merged headers


@dataclass
class ChunkMetadata:
    """
    [PURPOSE]
    Data model storing comprehensive metadata about a document chunk. This dataclass contains
    information about the chunk's source location (pages, header), size metrics (character
    count, word count), hierarchical relationships (parent/child headers), and section numbering.
    The metadata enables efficient chunk organization, search, and citation. Section numbers
    are particularly important for document navigation and referencing.
    
    [FIELD DESCRIPTIONS]
    - header_text: String containing the text of the section header for this chunk. This is
      the title/heading that identifies the chunk's topic. Used for display, search, and
      hierarchy building. Required field.
    - header_level: Integer representing the hierarchy level of the header (1 = top level,
      2 = subsection, etc.). Indicates the chunk's position in the document hierarchy.
      Used for organizing chunks and building table of contents. Required field.
    - font_size: Float representing the font size of the header in points (e.g., 14.0, 16.0).
      Extracted from the header's FontInfo. Used for font analysis and header identification.
      Required field.
    - font_name: String containing the font family name of the header (e.g., "Arial", "Calibri").
      Extracted from the header's FontInfo. Used for document analysis. Required field.
    - page: Integer representing the primary page number where the header appears (1-indexed).
      This is typically the same as start_page. Used for quick reference. Required field.
    - start_page: Integer representing the first page of the chunk's content (1-indexed).
      The chunk content may span multiple pages. Used for page range tracking and citation.
      Required field.
    - end_page: Integer representing the last page of the chunk's content (1-indexed). The chunk
      content spans from start_page to end_page. Used for page range tracking and citation.
      Required field.
    - char_count: Integer representing the number of characters in the chunk's content.
      Defaults to 0 if not provided. Calculated during chunk creation. Used for size metrics
      and statistics. Optional field with default.
    - word_count: Integer representing the number of words in the chunk's content. Defaults to 0
      if not provided. Calculated by splitting content on whitespace. Used for size metrics and
      statistics. Optional field with default.
    - parent_header: Optional string containing the text of the parent header. If this chunk
      is a subsection, parent_header contains the higher-level section's header text. If this
      is a root section, parent_header is None. Used for hierarchy navigation. Optional field.
    - child_headers: List of strings containing the texts of child headers (subsections under
      this chunk). The list is maintained in document order. Used for hierarchy navigation and
      building nested structures. Defaults to empty list if not provided.
    - section_number: Optional string containing the hierarchical section number (e.g., "1.1",
      "1.2.3", "2.5.1.4"). The section number is computed based on the chunk's position in the
      hierarchy and represents its location in the document structure. Used for citation and
      navigation. Computed during chunk processing. Optional field.
    
    [DATA FLOW]
    When instantiated:
    1. Required fields (header_text, header_level, font_size, font_name, page, start_page, end_page)
       must be provided.
    2. Optional fields (char_count, word_count, parent_header, child_headers, section_number)
       use defaults if not provided.
    3. If child_headers is None, it's initialized as empty list via field(default_factory=list).
    4. Python's dataclass mechanism automatically generates __init__, __repr__, __eq__ methods.
    5. Section numbers are typically computed later by _precompute_section_numbers() and assigned
       to chunk.metadata.section_number.
    
    [USAGE]
    Created by DocumentChunker._create_chunks_from_headers() when creating DocumentChunk objects.
    The metadata is attached to each chunk and used throughout the processing pipeline for
    organization, search, and output generation.
    
    [SIDE EFFECTS]
    - No side effects. This is a pure data container.
    - No validation or transformation occurs during instantiation (dataclass default behavior).
    """
    header_text: str  # Text of the section header
    header_level: int  # Hierarchy level (1 = root, 2 = subsection, etc.)
    font_size: float  # Font size of header in points
    font_name: str  # Font family name of header
    page: int  # Primary page number (typically same as start_page)
    start_page: int  # First page of chunk content (1-indexed)
    end_page: int  # Last page of chunk content (1-indexed)
    char_count: int = 0  # Number of characters in content
    word_count: int = 0  # Number of words in content
    parent_header: Optional[str] = None  # Text of parent header (None if root)
    child_headers: List[str] = field(default_factory=list)  # List of child header texts
    section_number: Optional[str] = None  # Hierarchical section number (e.g., "1.2.3")


@dataclass
class DocumentChunk:
    """
    [PURPOSE]
    Data model representing a logical section of a document that has been extracted and chunked.
    This is the primary data structure for document chunks in the RAG system. Each DocumentChunk
    contains the header text, full content, and comprehensive metadata. Chunks can be queried,
    searched, and processed independently. The chunk_id provides a unique identifier for
    referencing and storage.
    
    [FIELD DESCRIPTIONS]
    - chunk_id: String containing a unique identifier for the chunk (e.g., "introduction_001",
      "api_reference_042"). The ID is generated from the header text and index, making it
      meaningful and human-readable. Used for chunk lookup, storage, and referencing.
      Required field.
    - header: String containing the section header text. This is the title/heading that
      identifies what the chunk is about. Typically matches metadata.header_text. Used for
      display, search, and chunk identification. Required field.
    - content: String containing the full text content of the chunk. This includes all text
      elements under the header until the next header. The content may span multiple pages
      and contains the substantive information from that section. Required field.
    - metadata: ChunkMetadata object containing comprehensive metadata about the chunk (pages,
      font info, hierarchy, section number, etc.). This metadata enables efficient organization,
      search, and citation. Required field.
    - chunk_type: String specifying the type of chunking method used to create this chunk.
      Defaults to "font_based" (indicating font-size-based chunking). Other possible values
      might include "semantic", "paragraph", etc. Used for tracking chunking methodology.
      Optional field with default.
    
    [DATA FLOW]
    When instantiated:
    1. All required fields (chunk_id, header, content, metadata) must be provided.
    2. chunk_type defaults to "font_based" if not provided.
    3. Python's dataclass mechanism automatically generates __init__, __repr__, __eq__ methods.
    4. The chunk can be serialized to JSON for storage using serialize_chunk().
    
    [USAGE]
    Created by DocumentChunker._create_chunks_from_headers() when processing text elements and
    headers. DocumentChunk objects are stored in dictionaries keyed by chunk_id, saved to disk
    as JSON files, and used by the DeepAgentic framework for information retrieval.
    
    [SIDE EFFECTS]
    - No side effects. This is a pure data container.
    - No validation or transformation occurs during instantiation (dataclass default behavior).
    """
    chunk_id: str  # Unique identifier for the chunk
    header: str  # Section header text
    content: str  # Full text content of the chunk
    metadata: ChunkMetadata  # Comprehensive metadata (pages, hierarchy, section number, etc.)
    chunk_type: str = "font_based"  # Type of chunking method used


@dataclass
class ProcessingStats:
    """
    [PURPOSE]
    Data model for tracking statistics and metrics from document processing operations. This
    dataclass collects quantitative information about the processing pipeline (number of chunks
    created, headers detected, elements processed, etc.) for monitoring, debugging, and reporting
    purposes. The statistics help users understand the scale and results of document processing.
    
    [FIELD DESCRIPTIONS]
    - total_chunks: Integer representing the total number of document chunks created during
      processing. This is the primary metric indicating how many logical sections were extracted
      from the document. Defaults to 0. Updated during chunk creation.
    - total_headers: Integer representing the number of headers detected in the document.
      Headers are identified based on font size analysis and validation. This count may be
      less than total_chunks if some chunks don't have headers. Defaults to 0.
    - total_elements: Integer representing the total number of text elements processed from the
      PDF. This includes all lines of text extracted from all pages. Used for understanding
      document size. Defaults to 0.
    - processing_time: Float representing the time taken to process the document in seconds.
      This includes extraction, font analysis, chunking, and saving operations. Used for
      performance monitoring. Defaults to 0.0.
    - font_sizes_analyzed: Integer representing the number of unique font sizes found in the
      document. This metric helps understand font diversity and complexity. Extracted from
      FontAnalysis.font_distribution. Defaults to 0.
    
    [DATA FLOW]
    When instantiated:
    1. All fields have default values (0 or 0.0), so the dataclass can be created empty.
    2. Fields are typically populated during or after document processing by calling
       _generate_stats() or similar methods.
    3. Python's dataclass mechanism automatically generates __init__, __repr__, __eq__ methods.
    
    [USAGE]
    Created by DocumentProcessor._generate_stats() after document processing completes. The
    stats object is included in processing results and can be used for reporting and monitoring.
    
    [SIDE EFFECTS]
    - No side effects. This is a pure data container.
    - No validation or transformation occurs during instantiation (dataclass default behavior).
    """
    total_chunks: int = 0  # Number of chunks created
    total_headers: int = 0  # Number of headers detected
    total_elements: int = 0  # Total text elements processed
    processing_time: float = 0.0  # Time taken in seconds
    font_sizes_analyzed: int = 0  # Number of unique font sizes found


@dataclass
class FontAnalysis:
    """
    [PURPOSE]
    Data model containing the results of font usage pattern analysis in a document. This
    dataclass stores font distribution statistics, header level mappings, and body text
    identification. The analysis is performed by FontAnalyzer to identify which font sizes
    represent headers versus body text, enabling intelligent chunking based on document
    structure. The header_levels mapping is particularly important as it links font sizes to
    hierarchy levels for chunk creation.
    
    [FIELD DESCRIPTIONS]
    - font_distribution: Dictionary mapping font size (float) to occurrence count (int).
      This shows how frequently each font size appears in the document. Used to identify
      the most common size (body text) and rare sizes (likely headers). Defaults to empty dict.
    - header_levels: Dictionary mapping font size (float) to placeholder header level (int).
      Font sizes identified as headers are mapped to level 1 initially. Actual hierarchy levels
      are determined later in DocumentChunker based on font size relationships. Only contains
      header sizes, not body text. Defaults to empty dict.
    - body_text_size: Float representing the most common font size in the document, assumed
      to be body text. This is identified as the font size with the highest occurrence count.
      Used as a baseline for header detection (headers are larger than body text). Defaults to 0.0.

    [DATA FLOW]
    When instantiated:
    1. All fields have default values (empty dicts or 0.0), so the dataclass can be created empty.
    2. Fields are populated by FontAnalyzer.analyze_fonts() during document processing.
    3. font_distribution is built from counting font sizes across all text elements.
    4. body_text_size is identified as the most common font size.
    5. header_levels is built by identifying rare, large font sizes (frequency < threshold,
      size > body_text_size).
    6. Python's dataclass mechanism automatically generates __init__, __repr__, __eq__ methods.
    
    [USAGE]
    Created by FontAnalyzer.analyze_fonts() and passed to DocumentChunker.create_chunks() to
    guide header identification and chunk creation. The analysis results are also saved to
    disk for reference and debugging.
    
    [SIDE EFFECTS]
    - No side effects. This is a pure data container.
    - No validation or transformation occurs during instantiation (dataclass default behavior).
    """
    font_distribution: Dict[float, int] = field(default_factory=dict)  # Font size -> occurrence count
    header_levels: Dict[float, int] = field(default_factory=dict)  # Font size -> header level (placeholder)
    body_text_size: float = 0.0  # Most common font size (assumed body text)


@dataclass
class HierarchyStructure:
    """
    [PURPOSE]
    Data model representing the hierarchical organization of document sections. This dataclass
    tracks parent-child relationships between headers, enabling efficient navigation of the
    document structure. The hierarchy is built using a stack-based algorithm that processes
    headers in document order and establishes relationships based on level comparisons. This
    structure is essential for section numbering, chunk organization, and document index generation.
    
    [FIELD DESCRIPTIONS]
    - levels: Dictionary mapping header level (int) to a list of header texts (List[str]) at
      that level. For example, levels[1] contains all top-level headers, levels[2] contains
      all second-level headers, etc. Used for organizing headers by hierarchy depth and building
      level-based indexes. Defaults to empty dict.
    - parent_child: Dictionary mapping child header text (str) to parent header text (str).
      This establishes the parent-child relationships: if "1.2.3 Subsection" is a child of
      "1.2 Section", then parent_child["1.2.3 Subsection"] = "1.2 Section". Used for traversing
      up the hierarchy and computing section numbers. Defaults to empty dict.
    - root_headers: List of strings containing the texts of top-level headers (headers with no
      parent). These are the root sections of the document (level 1 headers that don't have a
      parent in the parent_child mapping). Used for identifying document structure and building
      the outline tree. Defaults to empty list.
    
    [DATA FLOW]
    When instantiated:
    1. All fields have default values (empty dicts or empty list), so the dataclass can be
       created empty.
    2. Fields are populated by DocumentChunker._build_hierarchy() using a stack-based algorithm:
       a. Process headers in document order.
       b. Maintain a stack of current parent chain.
       c. For each header, pop stack until finding a parent with lower level.
       d. Assign parent-child relationship.
       e. Build levels mapping by grouping headers by level.
       f. Identify root headers (those with no parent).
    3. Python's dataclass mechanism automatically generates __init__, __repr__, __eq__ methods.
    
    [USAGE]
    Created by DocumentChunker._build_hierarchy() and used throughout the processing pipeline
    for section numbering, chunk organization, and document index generation. The hierarchy
    structure is saved to disk for later reference.
    
    [SIDE EFFECTS]
    - No side effects. This is a pure data container.
    - No validation or transformation occurs during instantiation (dataclass default behavior).
    """
    levels: Dict[int, List[str]] = field(default_factory=dict)  # Header level -> list of header texts
    parent_child: Dict[str, str] = field(default_factory=dict)  # Child header text -> parent header text
    root_headers: List[str] = field(default_factory=list)  # List of top-level header texts (no parent)
