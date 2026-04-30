#!/usr/bin/env python3
"""
Document chunking functionality.

Creates intelligent chunks with hierarchy and context preservation,
enabling effective retrieval-augmented generation from PDF documents.

Example:
    >>> from lib.chunker import DocumentChunker
    >>> from lib.extractor import DocumentExtractor
    >>> from lib.analyzer import FontAnalyzer
    >>> extractor = DocumentExtractor()
    >>> analyzer = FontAnalyzer()
    >>> chunker = DocumentChunker()
    >>> elements = extractor.extract_text_elements("document.pdf")
    >>> font_analysis = analyzer.analyze_fonts(elements)
    >>> chunks, hierarchy = chunker.create_chunks(elements, font_analysis)
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from .models import (
    ChunkMetadata,
    DocumentChunk,
    FontAnalysis,
    HeaderInfo,
    HierarchyStructure,
    TextElement,
)

logger = logging.getLogger(__name__)

try:
    from config.constants import (
        FONT_SIZE_TOLERANCE_FOR_MERGE,
        LEFT_MARGIN_THRESHOLD,
        MAX_HEADER_TEXT_LENGTH,
        MAX_VERTICAL_GAP_FOR_MERGE,
        MIN_FONT_SIZE_THRESHOLD,
        MIN_HEADER_TEXT_LENGTH,
    )
except ImportError:
    MAX_HEADER_TEXT_LENGTH = 150
    MIN_FONT_SIZE_THRESHOLD = 8.0
    LEFT_MARGIN_THRESHOLD = 100
    MAX_VERTICAL_GAP_FOR_MERGE = 30.0
    FONT_SIZE_TOLERANCE_FOR_MERGE = 0.5
    MIN_HEADER_TEXT_LENGTH = 2

class DocumentChunker:
    """
    Handles document chunking with hierarchy and context preservation.
    
    Creates logical document chunks based on identified headers,
    preserving the hierarchical structure for better context.
    
    Example:
        >>> chunker = DocumentChunker()
        >>> chunks, hierarchy = chunker.create_chunks(elements, font_analysis)
        >>> print(f"Created {len(chunks)} chunks")
    """
    
    def __init__(self) -> None:
        """Initialize the chunker. Headers and header-to-chunk mapping are populated by create_chunks()."""
        pass

    def create_chunks(
        self, text_elements: List[TextElement], font_analysis: FontAnalysis
    ) -> Tuple[Dict[str, DocumentChunk], HierarchyStructure, Dict[tuple, DocumentChunk]]:
        """
        Create document chunks from text elements: identify headers, build hierarchy, extract content per header.

        Args:
            text_elements: Text elements from the extractor (one per line).
            font_analysis: Font analysis result with header_levels and body_text_size.

        Returns:
            Tuple of (chunks dict keyed by chunk_id, hierarchy structure, header_key_to_chunk_map).

        Example:
            >>> chunks, hierarchy, h2c = chunker.create_chunks(elements, font_analysis)
            >>> len(chunks)
        """
        # Step 1: Identify headers (simple font-size based)
        headers = self._identify_headers(text_elements, font_analysis)
        
        # Step 2: Build hierarchy based on font size relationships
        headers = self._build_font_size_hierarchy(headers)
        
        # Step 3: Build hierarchy structure using adjusted levels
        hierarchy = self._build_hierarchy(headers)
        
        # Store headers for later use (stack-based hierarchy with parent/children references)
        self.headers = headers
        
        # Step 4: Create chunks and header-to-chunk mapping
        chunks, header_key_to_chunk_map = self._create_chunks_from_headers(text_elements, headers)
        
        # Store mapping for later use
        self.header_key_to_chunk_map = header_key_to_chunk_map
        
        logger.info(f"Created {len(chunks)} chunks with {len(headers)} headers")
        return chunks, hierarchy, header_key_to_chunk_map
    
    def _identify_headers(self, text_elements: List[TextElement], font_analysis: FontAnalysis) -> List[HeaderInfo]:
        """
        Identify headers from text elements: by font size, merge wrapped lines, then validate.

        Args:
            text_elements: All text elements from the document.
            font_analysis: Font analysis with header_levels (which sizes are headers).

        Returns:
            List of validated HeaderInfo objects.
        """
        # Step 1: Identify all header candidates by font size only (no validation yet)
        header_candidates: List[HeaderInfo] = []
        header_levels = font_analysis.header_levels

        for element_index, element in enumerate(text_elements):
            # Font size is already rounded in FontInfo object during font analysis
            font_size = element.font.size
            if font_size not in header_levels:
                continue

            level = header_levels[font_size]

            header = HeaderInfo(
                text=element.text,
                level=level,
                font=element.font,
                page=element.page,
                position=element_index
            )
            header_candidates.append(header)

        header_candidates.sort(key=lambda h: (h.page, h.position))

        # Step 2: Merge consecutive headers that are part of the same line
        merged_headers = self._merge_consecutive_headers(header_candidates, text_elements)

        # Step 3: Validate merged headers (content quality, length, position)
        validated_headers: List[HeaderInfo] = []
        for header in merged_headers:
            # Build a proxy TextElement for validation (uses first segment's bbox)
            bbox = (
                text_elements[header.position].bbox
                if header.position < len(text_elements)
                else (0.0, 0.0, 0.0, 0.0)
            )
            proxy_element = TextElement(
                text=header.text,
                font=header.font,
                page=header.page,
                bbox=bbox,
                line_number=header.position,
            )
            if self._validate_header_candidate(proxy_element):
                validated_headers.append(header)

        return validated_headers
    
    def _validate_header_candidate(self, element: TextElement) -> bool:
        """
        Return True if the text element is a valid header (content quality, length, left margin).

        Filters out page numbers, indices, punctuation-only, and text too far from left margin.

        Args:
            element: TextElement to validate as a header candidate.

        Returns:
            True if valid header, False otherwise.
        """
        text = element.text.strip()
        
        # Filter empty or too-short text
        if not text or len(text) < MIN_HEADER_TEXT_LENGTH:
            return False
        
        # Filter text that is too long
        if len(text) > MAX_HEADER_TEXT_LENGTH:
            return False
        
        # Additional check: Headers with multiple sentences likely contain merged content
        if '.' in text:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 2:
                # More than 2 sentences suggests merged content
                return False

        if text.endswith('.'):
            return False
        
        # Filter numeric-only content (page numbers, indices, etc.)
        if re.match(r'^[\d\s\-\.]+$', text):
            return False
        
        # Filter punctuation-only or special character content
        if re.match(r'^[\W\s]+$', text):
            return False
        
        # Filter strings that are just whitespace with special chars
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        # Check x position (indentation) - headers typically start near left margin
        x_position = element.bbox[0]
        # if x_position > LEFT_MARGIN_THRESHOLD:
        #     return False
        
        # Safety check: ensure minimum readable font size
        if element.font.size < MIN_FONT_SIZE_THRESHOLD:
            return False
        
        return True
    
    def _merge_consecutive_headers(
        self, headers: List[HeaderInfo], text_elements: List[TextElement]
    ) -> List[HeaderInfo]:
        """
        Merges consecutive header candidates that are parts of a single wrapped header line.
        In PDFs, long headers often wrap across multiple lines, and each line may be detected
        as a separate header candidate. This method groups such consecutive candidates and
        combines them into a single HeaderInfo object with merged text. The merging logic
        checks for same page, same level, similar font size, close vertical proximity, and
        semantic coherence to ensure only legitimate wrapped headers are merged.
        """
        if not headers:
            return headers
        
        merged: List[HeaderInfo] = []
        current_group: List[HeaderInfo] = [headers[0]]
        
        for i in range(1, len(headers)):
            current_header = headers[i]
            prev_header = current_group[-1]
            
            # Check if headers should be merged
            if self._should_merge_headers(prev_header, current_header, text_elements):
                current_group.append(current_header)
            else:
                # Finalize current group and start new one
                merged.append(self._create_merged_header(current_group))
                current_group = [current_header]
        
        # Finalize last group
        if current_group:
            merged.append(self._create_merged_header(current_group))
        
        logger.debug(f"Merged {len(headers)} header candidates into {len(merged)} headers")
        return merged
    
    def _should_merge_headers(
        self, prev_header: HeaderInfo, curr_header: HeaderInfo, text_elements: List[TextElement]
    ) -> bool:
        """
        Determines whether two consecutive headers should be merged into a single header.
        Checks spatial proximity (same page, close vertical position), visual similarity
        (same level, similar font size), length constraints, and sentence boundaries
        that suggest separate sections.
        """
        # Must be on the same page
        if prev_header.page != curr_header.page:
            return False
        
        # Must have the same level
        if prev_header.level != curr_header.level:
            return False
        
        # Check font size similarity
        font_size_diff = abs(prev_header.font.size - curr_header.font.size)
        if font_size_diff > FONT_SIZE_TOLERANCE_FOR_MERGE:
            return False
        
        # Check if positions are consecutive or nearly consecutive
        position_gap = curr_header.position - prev_header.position
        if position_gap > 2:  # Allow at most one element between them
            return False
        
        # Check vertical proximity using bounding boxes
        if prev_header.position < len(text_elements) and curr_header.position < len(text_elements):
            prev_bbox = text_elements[prev_header.position].bbox
            curr_bbox = text_elements[curr_header.position].bbox
            
            # Vertical gap: top of current - bottom of previous
            vertical_gap = curr_bbox[1] - prev_bbox[3]
            
            if vertical_gap > MAX_VERTICAL_GAP_FOR_MERGE:
                return False
        
        # Check combined length
        combined_text = prev_header.text + " " + curr_header.text
        if len(combined_text) > MAX_HEADER_TEXT_LENGTH:
            return False
        
        # Check for sentence boundaries that suggest separate topics
        # If combined text has multiple sentences with substantial content, likely separate sections
        sentences = [s.strip() for s in combined_text.split('.') if s.strip()]
        if len(sentences) > 2:
            # Check if later sentences are substantial (not just abbreviations or short phrases)
            for i in range(1, len(sentences)):
                if len(sentences[i]) > 15:  # Substantial sentence
                    # Check if sentence starts with capital letter (likely new topic)
                    if sentences[i][0].isupper():
                        logger.debug(
                            f"Not merging headers: '{prev_header.text}' and '{curr_header.text}' - "
                            f"detected sentence boundary suggesting separate topics"
                        )
                        return False
        
        return True
    
    def _create_merged_header(self, header_group: List[HeaderInfo]) -> HeaderInfo:
        """
        Create one HeaderInfo from a group of headers (e.g. wrapped header lines). 
        Uses first header's properties.

        Args:
            header_group: List of headers to merge (at least one; typically 2-3 for one wrapped line).

        Returns:
            Single HeaderInfo with combined text; unchanged if group has only one header.
        """
        if len(header_group) == 1:
            return header_group[0]
        
        # Combine text from all headers
        combined_text = " ".join(h.text for h in header_group)
        
        # Use first header's properties for the merged header
        first_header = header_group[0]
        last_header = header_group[-1]
        
        # Collect original positions
        original_positions = [h.position for h in header_group]
        
        merged_header = HeaderInfo(
            text=combined_text,
            level=first_header.level,
            font=first_header.font,
            page=first_header.page,
            position=first_header.position,
            is_merged=True,
            original_positions=original_positions,
            end_position=last_header.position
        )
        
        logger.debug(f"Merged {len(header_group)} headers: '{combined_text[:50]}...'")
        return merged_header
    
    def _build_font_size_hierarchy(self, headers: List[HeaderInfo]) -> List[HeaderInfo]:
        """
        Assign hierarchy levels to headers from font size: larger = parent, same = sibling. Modifies headers in-place.

        Args:
            headers: List of HeaderInfo sorted by document position (page, position).

        Returns:
            Same list with header.level set (1 = root, higher = deeper child).
        """
        if not headers:
            return headers
        
        # Process headers in document order
        for i, header in enumerate(headers):
            if i == 0:
                # First header is always root (Level 1)
                header.level = 1
                continue
            
            # Find most recent preceding header with larger or equal font size
            parent = None
            same_size_header = None
            
            for prev_index in range(i - 1, -1, -1):
                prev_header = headers[prev_index]
                if prev_header.font.size > header.font.size:
                    parent = prev_header
                    break
                elif prev_header.font.size == header.font.size and same_size_header is None:
                    # Track first header with same font size for sibling relationship
                    same_size_header = prev_header
            
            if parent:
                # Parent has larger font size, so this is a child
                # Font sizes are integers (from round()), so any difference means child relationship
                header.level = parent.level + 1
            elif same_size_header:
                # Same font size as previous header = sibling (same level)
                header.level = same_size_header.level
            else:
                # No larger or same font found - check if it's larger than previous
                if i > 0 and header.font.size > headers[i-1].font.size:
                    # Larger font = new root section
                    header.level = 1
                else:
                    # Same or smaller, but no parent = root
                    header.level = 1
        
        return headers
    
    def _build_hierarchy(self, headers: List[HeaderInfo]) -> HierarchyStructure:
        """
        Build HierarchyStructure from headers using a stack: levels, parent_child map, root_headers. Modifies headers in-place (parent, children).

        Args:
            headers: List of HeaderInfo sorted by document order, with level already set.

        Returns:
            HierarchyStructure with levels, parent_child, and root_headers.
        """
        hierarchy = HierarchyStructure()
        hierarchy.levels = defaultdict(list)
        
        # Build level mapping
        for header in headers:
            hierarchy.levels[header.level].append(header.text)
        
        # Stack-based parent chain: stack[i] = parent at level i+1
        # As we process headers in order, maintain current parent chain
        # Stack invariant: stack is always sorted by level (ascending), representing current parent chain
        parent_stack: List[HeaderInfo] = []
        
        for header in headers:
            # Pop headers from stack until we find one with lower level (or stack is empty)
            # This maintains the invariant: stack[-1] is the immediate parent (if exists)
            while parent_stack and parent_stack[-1].level >= header.level:
                parent_stack.pop()
            
            # Assign parent (if stack not empty, top of stack is the parent)
            if parent_stack:
                parent = parent_stack[-1]
                header.parent = parent
                parent.children.append(header)
                hierarchy.parent_child[header.text] = parent.text
            else:
                # No parent = root header
                hierarchy.root_headers.append(header.text)
            
            # Push current header onto stack (it becomes potential parent for future children)
            parent_stack.append(header)
        
        return hierarchy
    
    def _create_chunks_from_headers(
        self, text_elements: List[TextElement], headers: List[HeaderInfo]
    ) -> Tuple[Dict[str, DocumentChunk], Dict[tuple, DocumentChunk]]:
        """
        Build DocumentChunk for each header: extract content until next header, create chunk ID and metadata, 1:1 header-to-chunk map.

        Args:
            text_elements: All text elements in the document.
            headers: Identified headers with parent/children set.

        Returns:
            Tuple of (chunk_id -> DocumentChunk, (text, page, position) -> DocumentChunk).
        """
        chunks: Dict[str, DocumentChunk] = {}
        header_key_to_chunk_map: Dict[tuple, DocumentChunk] = {}  # Key: (text, page, position) tuple
        
        for header_index, header in enumerate(headers):
            chunk_id = f"chunk_{header_index + 1:03d}"
            
            remaining_headers = headers[header_index + 1:] if header_index + 1 < len(headers) else []
            content_elements = self._extract_content_under_header(text_elements, header, remaining_headers)
            
            content_parts = [element.text for element in content_elements]
            content_text = '\n'.join(content_parts)
            
            parent_text = header.parent.text if header.parent else None
            children_texts = [child.text for child in header.children]
            
            metadata = ChunkMetadata(
                header_text=header.text,
                header_level=header.level,
                font_size=header.font.size,
                font_name=header.font.name,
                page=header.page,
                start_page=header.page,
                end_page=self._find_header_end_page(text_elements, header, headers, header_index),
                char_count=len(content_text),
                word_count=len(content_text.split()),
                parent_header=parent_text,
                child_headers=children_texts
            )
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                header=header.text,
                content=content_text,
                metadata=metadata
            )
            
            chunks[chunk_id] = chunk
            # Use tuple key (text, page, position) to uniquely identify each header
            header_key = (header.text, header.page, header.position)
            header_key_to_chunk_map[header_key] = chunk
        
        return chunks, header_key_to_chunk_map
    
    def _extract_content_under_header(
        self, text_elements: List[TextElement], header: HeaderInfo, next_headers: List[HeaderInfo]
    ) -> List[TextElement]:
        """
        Return text elements between this header and the next (content under the header). 
        If none, return one element with header text.

        Args:
            text_elements: All document text elements.
            header: Header whose content to extract.
            next_headers: Headers after this one (first one defines end position).

        Returns:
            List of TextElement for this section; at least one (header text if no content).
        """
        content_elements: List[TextElement] = []
        end_position = next_headers[0].position if next_headers else len(text_elements)
        
        for element_index in range(header.position + 1, end_position):
            if element_index < len(text_elements):
                element = text_elements[element_index]
                content_elements.append(element)
        
        if not content_elements:
            header_content = TextElement(
                text=header.text,
                font=header.font,
                page=header.page,
                bbox=(0, 0, 0, 0),
                line_number=header.position,
            )
            content_elements.append(header_content)
        
        return content_elements
    
    def _find_header_end_page(
        self, text_elements: List[TextElement], header: HeaderInfo, all_headers: List[HeaderInfo], current_index: int
    ) -> int:
        """
        Return the last page number that contains content for this header (between header and next header). 1-indexed.

        Args:
            text_elements: All document text elements.
            header: Header whose content end page to find.
            all_headers: All headers (used to find next header).
            current_index: Index of this header in all_headers.

        Returns:
            Page number where this header's content ends (or header.page if no content).
        """
        next_header = all_headers[current_index + 1] if current_index + 1 < len(all_headers) else None
        end_position = next_header.position if next_header else len(text_elements)
        
        last_content_page = header.page
        for element_index in range(header.position + 1, end_position):
            if element_index < len(text_elements):
                element = text_elements[element_index]
                last_content_page = element.page
        
        return last_content_page
