"""
Storage module for organized chunk file management.

This module provides a ChunkStorage class and utility functions for saving document chunks with:
- Organized folder hierarchy
- Consolidated and individual chunk files
- Metadata files (hierarchy, font analysis, document index)
- Simple outline generation (without page/word metadata)
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from tqdm import tqdm

from .models import DocumentChunk, HierarchyStructure, HeaderInfo

logger = logging.getLogger(__name__)


try:
    from exceptions import StorageError
except ImportError:
    StorageError = OSError  # type: ignore[misc, assignment]



class ChunkStorage:
    """
    Handles storage operations for document chunks.
    
    Provides methods for saving chunks in organized structures with metadata.
    """
    
    def __init__(self):
        """Initialize the ChunkStorage handler. Stateless; no configuration required."""
        pass
    
    def serialize_chunk(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Serialize a DocumentChunk to a dictionary suitable for JSON.

        Args:
            chunk: DocumentChunk to serialize.

        Returns:
            Dictionary with chunk_id, header, content, chunk_type, metadata, and optional section_number.
        """
        chunk_dict = {
            'chunk_id': chunk.chunk_id,
            'header': chunk.header,
            'content': chunk.content,
            'chunk_type': chunk.chunk_type,
            'metadata': {
                'header_level': chunk.metadata.header_level,
                'font_size': chunk.metadata.font_size,
                'font_name': chunk.metadata.font_name,
                'page_range': {
                    'start': chunk.metadata.start_page,
                    'end': chunk.metadata.end_page
                },
                'content_stats': {
                    'char_count': chunk.metadata.char_count,
                    'word_count': chunk.metadata.word_count
                },
                'hierarchy': {
                    'parent': chunk.metadata.parent_header,
                    'children': chunk.metadata.child_headers
                }
            }
        }
        
        # Add section_number if it exists in metadata
        if chunk.metadata.section_number:
            chunk_dict['section_number'] = chunk.metadata.section_number
        
        return chunk_dict
    
    def build_document_overview(
        self,
        chunks: Dict[str, DocumentChunk],
        hierarchy: Optional[HierarchyStructure]
    ) -> Dict[str, Any]:
        """Build document overview with totals, TOC, top-level sections, and hierarchy.

        Args:
            chunks: Dictionary of document chunks.
            hierarchy: Document hierarchy structure.

        Returns:
            Dict with total_chunks, total_pages, total_words, total_characters,
            top_level_sections, table_of_contents, and hierarchy. Empty dict if no chunks.
        """
        if not chunks:
            return {}
        
        # Get all chunks sorted by position
        sorted_chunks = sorted(
            chunks.values(),
            key=lambda chunk: (chunk.metadata.start_page, chunk.metadata.header_level)
        )
        
        # Build table of contents
        table_of_contents = []
        for chunk in sorted_chunks:
            toc_entry = {
                'title': chunk.header,
                'level': chunk.metadata.header_level,
                'page': chunk.metadata.start_page,
                'chunk_id': chunk.chunk_id
            }
            table_of_contents.append(toc_entry)
        
        # Get top-level sections (level 1 headers)
        top_sections = [
            chunk.header for chunk in sorted_chunks 
            if chunk.metadata.header_level == 1
        ]
        
        # Calculate statistics
        total_pages = max(
            chunk.metadata.end_page for chunk in chunks.values()
        ) if chunks else 0

        total_words = sum(
            chunk.metadata.word_count for chunk in chunks.values()
        )

        total_chars = sum(
            chunk.metadata.char_count for chunk in chunks.values()
        )
        
        return {
            'total_chunks': len(chunks),
            'total_pages': total_pages,
            'total_words': total_words,
            'total_characters': total_chars,
            'top_level_sections': top_sections,
            'table_of_contents': table_of_contents,
            'hierarchy': {
                'root_headers': (
                    hierarchy.root_headers if hierarchy else []
                ),
                'levels': (
                    dict(hierarchy.levels) if hierarchy else {}
                )
            }
        }
    
    def sanitize_folder_name(self, name: str) -> str:
        """Create a filesystem-safe folder name from a section title (lowercase, alphanumeric + underscores, max 40 chars). Returns "misc" if empty.

        Args:
            name: Original section name.

        Returns:
            Sanitized folder name.
        """
        if not name:
            return "misc"
        
        # Convert to lowercase and replace spaces/special chars
        clean = name.lower().strip()
        clean = clean.replace(' ', '_')
        clean = clean.replace('-', '_')
        
        # Keep only alphanumeric and underscores
        allowed = set('abcdefghijklmnopqrstuvwxyz0123456789_')
        clean = ''.join(c for c in clean if c in allowed)
        
        # Remove consecutive underscores and trim
        while '__' in clean:
            clean = clean.replace('__', '_')
        clean = clean.strip('_')
        
        # Limit length
        if len(clean) > 40:
            clean = clean[:40].rstrip('_')
        
        return clean or "misc"
    
    def get_folder_for_chunk(
        self,
        chunk: DocumentChunk,
        hierarchy: Optional[HierarchyStructure],
        index_data: Dict[str, Any]
    ) -> str:
        """Return subfolder name for a chunk from its root-level header. Empty if hierarchy/index_data missing.

        Args:
            chunk: The chunk to categorize.
            hierarchy: Document hierarchy structure.
            index_data: Document index data (must be truthy along with hierarchy).

        Returns:
            Folder name, or empty string to save in root.
        """
        if not index_data or not hierarchy:
            return ""
        
        # Get the top-level parent for this chunk
        header = chunk.header
        parent = header
        
        # Walk up the hierarchy to find the root parent
        while parent in hierarchy.parent_child:
            parent = hierarchy.parent_child[parent]
        
        # Sanitize folder name
        folder_name = self.sanitize_folder_name(parent)
        return folder_name
    
    def save_chunk_with_new_id_fast(
        self,
        folder_path: str,
        new_chunk_id: str,
        chunk: DocumentChunk,
        section_number: str
    ) -> None:
        """Save a chunk to a JSON file with a pre-computed section number (no hierarchy traversal).

        Args:
            folder_path: Directory to save the chunk.
            new_chunk_id: Chunk ID used as filename (e.g. chunk_001).
            chunk: The chunk to save.
            section_number: Pre-computed section number (e.g. "1.2.3").
        """
        chunk_data = {
            'chunk_id': new_chunk_id,
            'section_number': section_number,
            'header': chunk.header,
            'content': chunk.content,
            'chunk_type': chunk.chunk_type,
            'metadata': {
                'header_level': chunk.metadata.header_level,
                'font_size': chunk.metadata.font_size,
                'font_name': chunk.metadata.font_name,
                'page_range': {
                    'start': chunk.metadata.start_page,
                    'end': chunk.metadata.end_page
                },
                'content_stats': {
                    'char_count': chunk.metadata.char_count,
                    'word_count': chunk.metadata.word_count
                },
                'hierarchy': {
                    'parent': chunk.metadata.parent_header,
                    'children': chunk.metadata.child_headers
                }
            }
        }
        
        file_path = os.path.join(folder_path, f"{new_chunk_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
    
    def save_chunks_with_folder_structure(
        self,
        document_dir: str,
        chunks: Dict[str, DocumentChunk],
        hierarchy: Optional[HierarchyStructure],
        index_data: Dict[str, Any],
        quiet: bool = False,
    ) -> None:
        """Save all chunks into a single 'chunk' subfolder with sequential chunk IDs (chunk_001, ...).

        Args:
            document_dir: Base directory for the document.
            chunks: Dictionary of document chunks.
            hierarchy: Document hierarchy structure (for section numbers).
            index_data: Document index data (unused; kept for signature compatibility).
            quiet: If True, disable progress bars.
        """
        # Section numbers are computed by generate_document_index() using unique header keys
        # (text, page, position) which correctly handles duplicate header texts
        
        chunk_folder = os.path.join(document_dir, "chunk")
        os.makedirs(chunk_folder, exist_ok=True)

        logger.info("  Sorting chunks...")
        sorted_chunks = sorted(
            chunks.items(),
            key=lambda x: (
                x[1].metadata.start_page,
                x[1].metadata.header_level,
                x[0],
            ),
        )

        logger.info("  Saving %s chunks to chunk/...", len(sorted_chunks))
        for chunk_counter, (chunk_id, chunk) in enumerate(tqdm(
            sorted_chunks,
            desc="Saving chunks",
            unit="chunk",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]',
            disable=quiet,
        ), start=1):
            new_chunk_id = f"chunk_{chunk_counter:03d}"
            # Use section number stored on chunk metadata (set by generate_document_index)
            section_number = chunk.metadata.section_number or ""
            self.save_chunk_with_new_id_fast(
                chunk_folder, new_chunk_id, chunk, section_number
            )
    
    def save_font_analysis_data(
        self,
        document_dir: str,
        font_analysis: Any,
    ) -> Optional[str]:
        """Save font analysis to font_analysis.json. Returns None if font_analysis is falsy.

        Args:
            document_dir: Directory to save the file.
            font_analysis: Font analysis object.

        Returns:
            Path to the saved file, or None.
        """
        if not font_analysis:
            return None

        file_path = os.path.join(document_dir, "font_analysis.json")
        with open(file_path, 'w', encoding='utf-8') as file_handle:
            json.dump(
                font_analysis.__dict__,
                file_handle,
                indent=2,
                default=str,
                ensure_ascii=False,
            )
        return file_path

    def save_document_index(
        self,
        document_dir: str,
        document_name: str,
        index_data: Dict[str, Any]
    ) -> Optional[str]:
        """Save document_index.json and outline.txt (cleaned outline). Returns None if no index_data or on error.

        Args:
            document_dir: Directory to save the files.
            document_name: Document name.
            index_data: Document index (must contain outline_text for outline.txt).

        Returns:
            Path to document_index.json, or None.
        """
        try:
            if not index_data:
                logger.warning("No index data provided")
                return None
            
            index_filename = "document_index.json"
            index_path = os.path.join(document_dir, index_filename)
            
            with open(index_path, 'w', encoding='utf-8') as file_handle:
                json.dump(index_data, file_handle, indent=2, ensure_ascii=False)
            
            logger.info(f"Document index saved to '{index_path}'")
            
            # Also save simple outline (without page/word metadata)
            outline_text = index_data.get('outline_text', '')
            if outline_text:
                clean_outline = generate_simple_outline(outline_text)
                outline_path = os.path.join(document_dir, "outline.txt")
                with open(outline_path, 'w', encoding='utf-8') as file_handle:
                    file_handle.write(clean_outline)
                logger.info(f"Simple outline saved to '{outline_path}'")
            
            return index_path
            
        except Exception as error:
            logger.warning(f"Failed to save document index: {error}")
            return None
    
    def save_organized_chunks(
        self, output_dir: str, document_name: str, chunks: Dict[str, DocumentChunk], hierarchy: Optional[HierarchyStructure],
        font_analysis: Any, index_data: Dict[str, Any], quiet: bool = False
    ) -> None:
        """Save chunks to chunk/ and metadata (font_analysis.json, document_index.json, outline.txt).

        Args:
            output_dir: Base output directory.
            document_name: Name for the document subfolder.
            chunks: Dictionary of document chunks.
            hierarchy: Document hierarchy structure.
            font_analysis: Font analysis object.
            index_data: Document index data.
            quiet: If True, disable progress bars.

        Raises:
            StorageError: If the output directory cannot be created or files cannot be written.
        """
        document_dir = os.path.join(output_dir, document_name)
        try:
            os.makedirs(document_dir, exist_ok=True)
        except OSError as e:
            logger.exception("Could not create output directory: %s", document_dir)
            raise StorageError(f"Cannot create output directory: {e}") from e
        logger.debug("Output directory: %s", document_dir)

        # Create chunk folder and save chunks
        self.save_chunks_with_folder_structure(document_dir, chunks, hierarchy, index_data, quiet=quiet)

        # Save font analysis (fixed filename)
        self.save_font_analysis_data(document_dir, font_analysis)

        # Save document index and outline
        self.save_document_index(document_dir, document_name, index_data)

        logger.info("Saved %s chunks to '%s/'", len(chunks), document_dir)


class DocumentIndexGenerator:
    """
    Generates comprehensive document indexes for RAG systems.
    
    Creates structured indexes with:
    - Outline text (human-readable hierarchical view)
    - Structured index (JSON for programmatic access)
    """
    
    def __init__(self):
        """Initialize the DocumentIndexGenerator. Stateless; no configuration required."""
        pass
    
    def generate_document_index(
        self, document_name: str, chunks: Dict[str, DocumentChunk], hierarchy: Optional[HierarchyStructure], 
        headers: Optional[List[HeaderInfo]] = None, header_key_to_chunk_map: Optional[Dict[tuple, DocumentChunk]] = None
    ) -> Dict[str, Any]:
        """
        Build document index with outline_text and structured_index (sections tree). 
        Returns {} if chunks or hierarchy missing.

        Args:
            document_name: Name of the document.
            chunks: Dictionary of document chunks.
            hierarchy: Document hierarchy structure.
            headers: List of headers.
            header_key_to_chunk_map: Dictionary of header keys to document chunks.

        Returns:
            Dictionary containing the document index.
        """
        if not chunks or not hierarchy:
            return {}
        
        # Build the hierarchical tree with section numbers using stack-based hierarchy
        outline_tree = self._build_outline_tree(chunks, hierarchy, headers, header_key_to_chunk_map)
        
        # Generate text outline
        outline_text = self._generate_outline_text(document_name, outline_tree)
        
        return {
            "document_name": document_name,
            "generated_at": datetime.now().isoformat(),
            "outline_text": outline_text,
            "structured_index": {
                "sections": outline_tree,  # Full tree for programmatic access
                "total_sections": self._count_sections(outline_tree),
                "max_depth": self._get_max_depth(outline_tree)
            }
        }
    
    def _build_outline_tree(self, chunks: Dict[str, DocumentChunk], hierarchy: HierarchyStructure, headers: Optional[List[HeaderInfo]] = None, header_key_to_chunk_map: Optional[Dict[tuple, DocumentChunk]] = None) -> List[Dict[str, Any]]:
        """Build hierarchical outline tree from headers and chunk map. Returns [] if hierarchy/chunks missing; raises ValueError if headers or header_key_to_chunk_map missing."""
        if not hierarchy or not chunks:
            return []
        
        # Require headers and mapping for clean stack-based approach
        if not headers or not header_key_to_chunk_map:
            raise ValueError("Headers and header_key_to_chunk_map required for stack-based hierarchy")
        
        # Find root headers (headers with no parent)
        root_headers = [h for h in headers if h.parent is None]
        
        # Sort root headers by document position (page, position)
        root_headers.sort(key=lambda h: (h.page, h.position))
        
        tree: List[Dict[str, Any]] = []
        section_counter = 0
        
        for root_header in root_headers:
            section_counter += 1
            section = self._build_section_node_from_header(
                header=root_header,
                header_key_to_chunk_map=header_key_to_chunk_map,
                section_number=str(section_counter),
                depth=1
            )
            tree.append(section)
        
        return tree
    
    def _build_section_node_from_header(self, header: HeaderInfo, header_key_to_chunk_map: Dict[tuple, DocumentChunk], section_number: str, depth: int) -> Dict[str, Any]:
        """Build a section dict (number, title, depth, chunk_id, page_start, page_end, word_count, summary, children) from a header; recurses for children."""
        # Direct lookup using tuple key - guaranteed to exist since mapping created during chunk creation
        header_key = (header.text, header.page, header.position)
        chunk = header_key_to_chunk_map[header_key]
        
        # Store the correct section number in chunk metadata
        # This fixes the issue where _precompute_section_numbers uses header text as key
        # which causes collisions for duplicate headers (e.g., multiple "Parameters" sections)
        chunk.metadata.section_number = section_number
        
        # Build section data
        section: Dict[str, Any] = {
            "number": section_number,
            "title": header.text,
            "depth": depth,
            "chunk_id": chunk.chunk_id,
            "page_start": chunk.metadata.start_page,
            "page_end": chunk.metadata.end_page,
            "word_count": chunk.metadata.word_count
        }
        
        # Generate brief summary
        content = chunk.content.strip()
        first_sentence_end = content.find('.')
        if first_sentence_end > 0 and first_sentence_end < 150:
            section["summary"] = content[:first_sentence_end + 1]
        else:
            section["summary"] = content[:100] + "..." if len(content) > 100 else content
        
        # Recursively build children using header.children (already in document order!)
        if header.children:
            section["children"] = []
            for child_index, child_header in enumerate(header.children):
                child_section = self._build_section_node_from_header(
                    header=child_header,
                    header_key_to_chunk_map=header_key_to_chunk_map,
                    section_number=f"{section_number}.{child_index + 1}",
                    depth=depth + 1
                )
                section["children"].append(child_section)
        
        return section
    
    def _generate_outline_text(self, document_name: str, tree: List[Dict[str, Any]]) -> str:
        """Generate human-readable outline text from section tree (indented, with page/word metadata)."""
        lines: List[str] = []
        lines.append(f"DOCUMENT INDEX: {document_name}")
        lines.append("=" * (16 + len(document_name)))
        lines.append("")
        
        def format_section(section: Dict[str, Any], indent: int = 0) -> None:
            """Recursively format sections with indentation."""
            prefix = "   " * indent
            number = section["number"]
            title = section["title"]
            
            # Build metadata string
            meta_parts: List[str] = []
            
            if "page_start" in section:
                if section.get("page_start") == section.get("page_end"):
                    meta_parts.append(f"Page {section['page_start']}")
                else:
                    meta_parts.append(f"Pages {section['page_start']}-{section['page_end']}")
            
            if "word_count" in section and section["word_count"] > 0:
                meta_parts.append(f"{section['word_count']} words")
            
            meta_str = f" [{', '.join(meta_parts)}]" if meta_parts else ""
            
            lines.append(f"{prefix}{number}. {title}{meta_str}")
            
            # Process children
            for child in section.get("children", []):
                format_section(child, indent + 1)
        
        for section in tree:
            format_section(section)
        
        return "\n".join(lines)
    
    def _count_sections(self, tree: List[Dict[str, Any]]) -> int:
        """Return total number of sections in the tree (root + all nested children)."""
        count = 0
        
        def count_recursive(sections: List[Dict[str, Any]]) -> None:
            nonlocal count
            for section in sections:
                count += 1
                count_recursive(section.get("children", []))
        
        count_recursive(tree)
        return count
    
    def _get_max_depth(self, tree: List[Dict[str, Any]]) -> int:
        """Return maximum nesting depth of the section tree (0 = root only)."""
        if not tree:
            return 0
        
        def get_depth(sections: List[Dict[str, Any]], current: int) -> int:
            if not sections:
                return current
            max_child_depth = current
            for section in sections:
                child_depth = get_depth(section.get("children", []), current + 1)
                max_child_depth = max(max_child_depth, child_depth)
            return max_child_depth
        
        return get_depth(tree, 0)


def generate_simple_outline(outline_text: str) -> str:
    """Remove page/word metadata (e.g. '[Page 1, 276 words]') from outline text; preserve structure and section numbers."""
    lines = outline_text.split('\n')
    clean_lines = []
    
    for line in lines:
        # Remove patterns like [Page 1, 276 words] or [Pages 1-2, 263 words]
        clean_line = re.sub(
            r'\s*\[Pages?\s+\d+(?:-\d+)?,\s*\d+\s*words?\]',
            '',
            line
        )
        clean_lines.append(clean_line)
    
    return '\n'.join(clean_lines)
