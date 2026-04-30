#!/usr/bin/env python3
"""
Font analysis functionality for document processing.

Analyzes fonts to identify headers using frequency-based detection.
Headers are identified as rare font sizes (compared to common body text).
Actual hierarchy levels are built in chunker.py based on header-to-header comparisons.
"""

import logging
from collections import Counter
from typing import Dict, List

from .models import FontAnalysis, TextElement

logger = logging.getLogger(__name__)

try:
    from config.constants import MAX_COMMON_FREQ
except ImportError:
    MAX_COMMON_FREQ = 0.2  # Headers are rare (<20% of document)


class FontAnalyzer:
    """
    Handles font analysis for document structure detection.
    
    Analyzes font usage patterns across a document to identify headers
    using frequency-based detection. Headers are identified as rare font
    sizes (compared to common body text). Actual hierarchy levels are
    built in chunker.py based on header-to-header font size comparisons.
    """
    
    def __init__(self) -> None:
        pass

    def analyze_fonts(self, text_elements: List[TextElement]) -> FontAnalysis:
        """
        Analyzes font usage patterns across all text elements in a document to identify headers
        and body text. This method performs frequency-based header detection by identifying
        rare font sizes (compared to common body text). It rounds font sizes to integers using
        round() for consistency, counts font size occurrences, identifies the most common
        size (body text), and maps header font sizes to placeholder levels. The analysis
        provides the foundation for intelligent chunking.

        Modifies each element's font.size in place (rounded to integer).

        Args:
            text_elements: List of TextElement objects from the extractor (one per line).

        Returns:
            FontAnalysis with font_distribution, header_levels, and body_text_size.
        """
        font_size_counts: Counter[float] = Counter()

        for element in text_elements:
            element.font.size = round(element.font.size)
            font_size_counts[element.font.size] += 1

        body_text_size = font_size_counts.most_common(1)[0][0] if font_size_counts else 0
        logger.debug("Font size distribution: %s elements, body size: %s pt", len(text_elements), body_text_size)
        header_levels = self._create_header_level_mapping(font_size_counts, body_text_size)
        font_analysis = FontAnalysis(font_distribution=dict(font_size_counts), 
            header_levels=header_levels, body_text_size=body_text_size
        )
        logger.info("Font analysis completed: %s unique font sizes, body text size: %spt", len(font_size_counts), body_text_size)
        return font_analysis
    
    def _create_header_level_mapping(self, font_size_counts: Counter[float], body_size: float) -> Dict[float, int]:
        """
        Identifies header font sizes using frequency-based detection. This method implements a
        simple one-pass algorithm that identifies headers as rare font sizes (appearing less
        frequently than a threshold) that are larger than body text. All identified header sizes
        are mapped to placeholder level 1 - actual hierarchy levels are determined later in
        DocumentChunker based on header-to-header font size comparisons. This method provides
        the initial header identification that enables subsequent hierarchy building.

        Args:
            font_size_counts: Count of each font size across the document.
            body_size: Most common font size (assumed body text); header sizes must be larger.

        Returns:
            Dict mapping font size (float) to placeholder header level (1).
        """
        if not font_size_counts or body_size <= 0:
            return {}
        
        header_levels: Dict[float, int] = {}
        total = sum(font_size_counts.values())
        
        for size, count in font_size_counts.items():
            if size <= body_size:
                continue
            
            frequency_percentage = count / total
            if frequency_percentage > MAX_COMMON_FREQ:
                continue
            
            header_levels[size] = 1
        
        header_count = len(header_levels)
        if header_count > 0:
            header_sizes = sorted(header_levels.keys(), reverse=True)
            logger.info("Header detection: %s header size(s) (body: %spt, headers: %s)", 
                header_count, body_size, ", ".join(f"{s}pt" for s in header_sizes)
            )
        else:
            logger.info("Header detection: No headers identified (body: %spt)", body_size)
        
        return header_levels
