"""
Tool for reading the outline.txt file for a document with stepwise display support.

Includes outline parsing helpers used exclusively by this tool.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from ..errors import DocumentNotFoundError
from ..logging_config import get_logger
from ._common import get_chunks_dir, _get_available_document_names

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Outline parsing helpers
# ---------------------------------------------------------------------------

def _parse_outline_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single outline line to extract section information.
    
    Args:
        line: A line from the outline file
        
    Returns:
        Dictionary with section info or None if line doesn't contain a section
    """
    # Pattern to match: optional whitespace, section number (digits with dots), period, title, optional metadata
    pattern = r'^(\s*)(\d+(?:\.\d+)*)\.\s+(.+?)(?:\s+\[.*?\])?$'
    match = re.match(pattern, line.strip())
    
    if not match:
        return None
    
    indentation = match.group(1)
    section_number = match.group(2)
    title_part = match.group(3)
    
    # Extract title (everything before metadata in brackets if present)
    title = title_part.split(' [')[0].strip()
    
    # Extract metadata if present
    metadata_match = re.search(r'\[(.*?)\]', line)
    metadata = metadata_match.group(1) if metadata_match else None
    
    # Calculate hierarchy level: count dots in section number
    level = section_number.count('.') + 1
    
    return {
        'section_number': section_number,
        'level': level,
        'title': title,
        'metadata': metadata,
        'indentation': indentation,
        'full_line': line.rstrip()
    }


def _count_child_headers(lines: List[Dict[str, Any]], parent_index: int) -> int:
    """
    Count child headers (Level 3+) under a parent section.
    
    Args:
        lines: List of parsed outline line dictionaries
        parent_index: Index of the parent section in the lines list
        
    Returns:
        Number of child headers (Level 3+) under the parent
    """
    if parent_index >= len(lines):
        return 0
    
    parent = lines[parent_index]
    parent_section = parent['section_number']
    parent_level = parent['level']
    
    # Children must be at least Level 3 and start with parent's section number + "."
    child_prefix = parent_section + "."
    count = 0
    
    for i in range(parent_index + 1, len(lines)):
        line = lines[i]
        line_section = line['section_number']
        line_level = line['level']
        
        # Stop if we hit a sibling (same or higher level that doesn't start with parent prefix)
        if line_level <= parent_level:
            break
        
        # Count all children (Level 3+) that start with parent prefix
        if line_section.startswith(child_prefix) and line_level >= 3:
            count += 1
        # Stop if we hit a section that doesn't start with parent prefix (sibling branch)
        elif not line_section.startswith(child_prefix):
            break
    
    return count


def _find_section_index(lines: List[Dict[str, Any]], section_number: str) -> Optional[int]:
    """
    Find the line index of a section number by exact match.
    
    Args:
        lines: List of parsed outline line dictionaries
        section_number: Section number to find (e.g., "1.2", "1.4.1")
        
    Returns:
        Index of the section or None if not found
    """
    for i, line in enumerate(lines):
        if line['section_number'] == section_number:
            return i
    return None


def _get_section_children(lines: List[Dict[str, Any]], section_index: int) -> List[Dict[str, Any]]:
    """
    Get all child lines under a section.
    
    Args:
        lines: List of parsed outline line dictionaries
        section_index: Index of the parent section
        
    Returns:
        List of child line dictionaries
    """
    if section_index >= len(lines):
        return []
    
    parent = lines[section_index]
    parent_section = parent['section_number']
    parent_level = parent['level']
    
    # Children must start with parent's section number + "."
    child_prefix = parent_section + "."
    children = []
    
    for i in range(section_index + 1, len(lines)):
        line = lines[i]
        line_section = line['section_number']
        line_level = line['level']
        
        # Stop if we hit a sibling (same or higher level)
        if line_level <= parent_level:
            break
        
        # Include all lines that start with parent prefix
        if line_section.startswith(child_prefix):
            children.append(line)
        else:
            # Stop if we hit a section that doesn't start with parent prefix
            break
    
    return children


def _get_immediate_children(lines: List[Dict[str, Any]], section_index: int) -> List[Dict[str, Any]]:
    """
    Get only direct (one-level-down) child lines under a section.
    
    Args:
        lines: List of parsed outline line dictionaries
        section_index: Index of the parent section
        
    Returns:
        List of immediate child line dictionaries (e.g. for "1.2" returns only 1.2.1, 1.2.2, not 1.2.1.1)
    """
    if section_index >= len(lines):
        return []
    
    parent = lines[section_index]
    parent_section = parent['section_number']
    parent_level = parent['level']
    child_prefix = parent_section + "."
    
    immediate = []
    for i in range(section_index + 1, len(lines)):
        line = lines[i]
        line_section = line['section_number']
        line_level = line['level']
        
        if line_level <= parent_level:
            break
        if not line_section.startswith(child_prefix):
            break
        # Immediate child: suffix after prefix has no dot (e.g. 1.2.3 -> suffix "3", not "3.1")
        suffix = line_section[len(child_prefix):]
        if "." not in suffix:
            immediate.append(line)
    
    return immediate


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
def read_outline(document_name: str, expand_section: Optional[str] = None) -> str:
    """
    Reads the outline.txt file for a document with stepwise display support.
    
    Purpose:
        Retrieves the human-readable table of contents (outline) for a document. This provides
        a formatted text representation of the document's hierarchical structure. Sub-levels
        are never shown directly in default or expansion mode; instead a clue is given
        (e.g. "N sub levels present"). Use expand_section to reveal the next level under a
        given section, or expand_section="0" for the full outline.
    
    When to use:
        - To get a human-readable table of contents for understanding document structure
        - To understand document hierarchy and section relationships
        - To find section locations and page numbers in a readable format
        - When you need to present document structure to the user in a readable way
        - For large documents: Use default mode for Level 1 only, then expand_section to drill down by section
    
    Args:
        document_name (str): Document folder name (e.g., "CrowdStrike", "CiscoDuo")
            This should match one of the document names returned by discover_documents().
            Example: "ZScalar"
        
        expand_section (str, optional): Controls the outline display mode:
            - None (default): Show only Level 1 headers; each with "(N sub levels present)" when it has descendants
            - "<section_number>": Show the specified section and its immediate (one-level) children only; each child with descendants gets "(N sub levels present)"
            - "0": Return the complete outline file as-is
            Examples: None, "1.4", "1.5.1", "0"
    
    Returns:
        str: Formatted outline string. The format depends on expand_section parameter:
        
        Default mode (expand_section=None):
            - Document header
            - Level 1 headers only (Level 2+ are not listed)
            - For each Level 1 that has descendants: "(N sub levels present)"
            - Example: "1. Duo Admin API (25 sub levels present)"
        
        Expansion mode (expand_section="<section_number>"):
            - The specified section header
            - Only the immediate next level under that section (one level of children)
            - For each of those children that has its own descendants: "(N sub levels present)"
            - Deeper levels are not shown; expand again by section number to drill further
        
        Full mode (expand_section="0"):
            - Complete outline.txt file content as-is
            - All sections at all levels
    
    Raises:
        FileNotFoundError: If outline.txt file does not exist for the specified document,
            or if the chunks directory is not initialized.
        ValueError: If expand_section is provided but the section number is not found in the outline.
    
    Usage examples:
        - User query: "Show me the table of contents" → read_outline("DocumentName") for Level 1 with sub-level counts
        - User query: "Show me what's under section 1.4" → read_outline("DocumentName", expand_section="1.4")
        - User query: "Show me the complete outline" → read_outline("DocumentName", expand_section="0")
        - Stepwise: read_outline(...) → then read_outline(..., expand_section="1") → then read_outline(..., expand_section="1.2") etc.
    
    Related tools:
        - Use discover_documents() first if you don't know the document name
        - Use read_document_index() if you need structured/programmatic access instead of text format
        - After identifying relevant sections from the outline, use get_chunks_by_section() to retrieve specific chunks
        - Use search_chunks() for keyword-based searches when section numbers are unknown
    """
    _chunks_dir = get_chunks_dir()
    if not _chunks_dir:
        return f"Error: Chunks directory \"{_chunks_dir}\" not initialized. Cannot read outline."
    
    outline_path = Path(os.path.join(_chunks_dir, document_name, "outline.txt"))
    if not outline_path.exists():
        available = _get_available_document_names()
        logger.warning("Outline not found", extra={"extra_fields": {"document_name": document_name, "available": available}})
        raise DocumentNotFoundError(
            f"Outline not found for document: '{document_name}'. Use an EXACT path from discover_documents().",
            available_documents=available,
        )
    
    start_time = time.time()
    with open(outline_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if expand_section == "0":
        elapsed = time.time() - start_time
        logger.info("read_outline full", extra={"extra_fields": {"elapsed": round(elapsed, 2)}})
        return ''.join(lines)
    
    # Parse all lines
    parsed_lines = []
    header_lines = []
    in_header = True
    
    for line in lines:
        # Collect header lines (DOCUMENT INDEX and separator)
        if in_header:
            if line.strip().startswith("DOCUMENT INDEX"):
                header_lines.append(line)
            elif line.strip().startswith("="):
                header_lines.append(line)
                in_header = False
            elif line.strip() == "":
                header_lines.append(line)
            else:
                in_header = False
        
        # Parse content lines
        if not in_header:
            parsed = _parse_outline_line(line)
            if parsed:
                parsed_lines.append(parsed)
    
    # Expansion mode: show the requested section and only its immediate (one-level-down) children.
    # For each child that has its own descendants, show "(N sub levels present)".
    if expand_section:
        section_index = _find_section_index(parsed_lines, expand_section)
        if section_index is None:
            return f"Section '{expand_section}' not found in this outline. Check the section number from the outline or use expand_section='0' to see the full outline."

        output_lines = header_lines.copy()
        parent = parsed_lines[section_index]
        output_lines.append(parent['full_line'] + '\n')
        immediate_children = _get_immediate_children(parsed_lines, section_index)
        for child in immediate_children:
            line_text = child['full_line'].rstrip()
            child_index = _find_section_index(parsed_lines, child['section_number'])
            if child_index is not None:
                descendant_count = len(_get_section_children(parsed_lines, child_index))
                if descendant_count > 0:
                    line_text = f"{line_text} ({descendant_count} sub levels present)"
            output_lines.append(line_text + '\n')
        elapsed = time.time() - start_time
        return ''.join(output_lines)

    # Default mode: show only Level 1; do not show Level 2+ directly. For each Level 1,
    # add "(N sub levels present)" when it has any descendants so the user can expand by section.
    output_lines = header_lines.copy()
    for i, line_info in enumerate(parsed_lines):
        if line_info['level'] != 1:
            continue
        line_text = line_info['full_line'].rstrip()
        descendant_count = len(_get_section_children(parsed_lines, i))
        if descendant_count > 0:
            line_text = f"{line_text} ({descendant_count} sub levels present)"
        output_lines.append(line_text + '\n')

    elapsed = time.time() - start_time
    return ''.join(output_lines)
