#!/usr/bin/env python3
"""
Core document processing functionality.

Main entry point for PDF document analysis and chunking, providing
a unified interface for the complete document processing pipeline.

Example:
    >>> from utils import DocumentProcessor
    >>> processor = DocumentProcessor()
    >>> result = processor.process_document("document.pdf")
    >>> if result['success']:
    ...     print(f"Created {result['total_chunks']} chunks")
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .analyzer import FontAnalyzer
from .chunker import DocumentChunker
from .extractor import DocumentExtractor
from .storage import ChunkStorage, DocumentIndexGenerator
from .models import (
    ChunkMetadata,
    DocumentChunk,
    FontAnalysis,
    HierarchyStructure,
    ProcessingStats,
    ProcessingResult,
)

logger = logging.getLogger(__name__)

# Optional config: use defaults if not available (e.g. in tests)
try:
    from config.settings import ChunkerSettings
except ImportError:
    ChunkerSettings = None  # type: ignore[misc, assignment]

try:
    from exceptions import (
        ChunkerFileNotFoundError,
        InvalidFileError,
        ExtractionError,
        ProcessingError,
        StorageError,
    )
except ImportError:
    # Fallback when config/exceptions not on path
    ChunkerFileNotFoundError = Exception  # type: ignore[misc, assignment]
    InvalidFileError = Exception  # type: ignore[misc, assignment]
    ExtractionError = Exception  # type: ignore[misc, assignment]
    ProcessingError = Exception  # type: ignore[misc, assignment]
    StorageError = Exception  # type: ignore[misc, assignment]


def _get_settings(settings: Optional[Any] = None) -> Any:
    """Return ChunkerSettings instance; use defaults if not available."""
    if settings is not None:
        return settings
    if ChunkerSettings is not None:
        return ChunkerSettings()
    # Minimal inline defaults when config not installed
    class Defaults:
        output_dir = Path("output")
        source_data_dir = Path("data")
    return Defaults()


class DocumentProcessor:
    """
    Main document processor for PDF analysis and chunking.

    Orchestrates the entire document processing pipeline:
    1. Extract text and font information from PDF
    2. Analyze fonts and identify headers
    3. Create intelligent chunks with hierarchy

    Attributes:
        settings: Optional ChunkerSettings instance.
        extractor: DocumentExtractor instance.
        analyzer: FontAnalyzer instance.
        chunker: DocumentChunker instance.
        chunks: Dictionary of processed document chunks.
        hierarchy: Document hierarchy structure.
        font_analysis: Font analysis results.
        stats: Processing statistics.
    """

    def __init__(self, settings: Optional[Any] = None) -> None:
        """
        Initialize the DocumentProcessor with optional settings and component instances.

        Args:
            settings: Optional ChunkerSettings (or object with output_dir, source_data_dir, max_file_size_mb).
                     If None, defaults are used.
        """
        self.settings = _get_settings(settings)
        self.extractor = DocumentExtractor()
        self.analyzer = FontAnalyzer()
        self.chunker = DocumentChunker()
        self.storage = ChunkStorage()
        self.index_generator = DocumentIndexGenerator()

        self.chunks: Dict[str, DocumentChunk] = {}
        self.hierarchy: Optional[HierarchyStructure] = None
        self.font_analysis: Optional[FontAnalysis] = None
        self.stats: Optional[ProcessingStats] = None

    def process_document(self, pdf_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None, quiet: bool = False) -> ProcessingResult:
        """
        Run the full document pipeline (extract, analyze, chunk, index, save) and return
        metadata plus output location.

        Args:
            pdf_path: Path to the PDF file. Must exist, be a file, and have a .pdf suffix.
            output_dir: Override output directory for chunks. Defaults to settings.output_dir.
            quiet: If True, reduce log output (e.g. for progress).

        Returns:
            A dict with success True and document_info, chunks_location, hierarchy,
            font_analysis, stats, total_chunks.

        Raises:
            ChunkerFileNotFoundError: PDF file does not exist.
            InvalidFileError: Path is not a file, not .pdf, or file too large.
            ExtractionError: PDF text extraction failed.
            ProcessingError: Chunking or analysis failed.
            StorageError: Saving chunks to disk failed.
        """
        resolved_pdf_path = Path(pdf_path).resolve()
        pdf_path_str = str(resolved_pdf_path)

        if not resolved_pdf_path.exists():
            logger.warning("PDF file not found: %s", pdf_path_str)
            raise ChunkerFileNotFoundError(f"PDF file not found: '{pdf_path_str}'")
        if not resolved_pdf_path.is_file():
            logger.warning("Path is not a file: %s", pdf_path_str)
            raise InvalidFileError(f"Path is not a file: '{pdf_path_str}'")
        if resolved_pdf_path.suffix.lower() != ".pdf":
            logger.warning("File is not a PDF: %s", pdf_path_str)
            raise InvalidFileError(f"File is not a PDF: '{pdf_path_str}'")

        chunks_base_dir = Path(output_dir) if output_dir else Path(self.settings.output_dir)
        chunks_base_dir = chunks_base_dir.resolve()
        # Validate output directory is writable (if it exists)
        if chunks_base_dir.exists() and not os.access(chunks_base_dir, os.W_OK):
            raise StorageError(f"No write permission for output directory: {chunks_base_dir}")

        source_data_dir = getattr(self.settings, "source_data_dir", Path("data"))
        source_data_dir = Path(source_data_dir)
        chunks_subfolder_path: Path = Path(resolved_pdf_path.stem)
        if source_data_dir.is_dir():
            try:
                data_root_path = source_data_dir.resolve()
                pdf_relative_to_data = resolved_pdf_path.relative_to(data_root_path)
                chunks_subfolder_path = pdf_relative_to_data.parent / pdf_relative_to_data.stem
            except (ValueError, TypeError):
                pass
        document_folder_name = str(chunks_subfolder_path)

        logger.info("Processing document: %s", pdf_path_str)

        try:
            logger.info("Step 1: Extracting text elements")
            text_elements = self.extractor.extract_text_elements(pdf_path_str)
        except (ExtractionError, InvalidFileError):
            raise
        except Exception as e:
            logger.exception("Unexpected extraction error for %s", pdf_path_str)
            raise ExtractionError(f"Failed to extract text: {e}") from e

        try:
            logger.info("Step 2: Analyzing font patterns")
            self.font_analysis = self.analyzer.analyze_fonts(text_elements)
        except Exception as e:
            logger.exception("Font analysis failed for %s", pdf_path_str)
            raise ProcessingError(f"Font analysis failed: {e}") from e

        try:
            logger.info("Step 3: Creating document chunks")
            self.chunks, self.hierarchy, self.header_key_to_chunk_map = self.chunker.create_chunks(text_elements, self.font_analysis)
        except Exception as e:
            logger.exception("Chunking failed for %s", pdf_path_str)
            raise ProcessingError(f"Chunking failed: {e}") from e

        logger.debug("Document processing: %s chunks created", len(self.chunks))
        self.stats = self._generate_stats()

        headers = getattr(self.chunker, "headers", None)
        try:
            index_data = self.index_generator.generate_document_index(document_folder_name, 
                self.chunks, self.hierarchy, headers, self.header_key_to_chunk_map,
            )
        except Exception as e:
            logger.exception("Index generation failed for %s", pdf_path_str)
            raise ProcessingError(f"Index generation failed: {e}") from e

        try:
            logger.info("Step 4: Saving chunks to disk")
            self.storage.save_organized_chunks(
                str(chunks_base_dir),
                document_folder_name,
                self.chunks,
                self.hierarchy,
                self.font_analysis,
                index_data,
                quiet=quiet,
            )
        except Exception as e:
            logger.exception("Saving chunks failed for %s", pdf_path_str)
            raise StorageError(f"Failed to save chunks: {e}") from e

        chunks_location = str(chunks_base_dir / chunks_subfolder_path)
        logger.info("Chunks saved to %s/", chunks_location)

        logger.info("Step 5: Generating document overview")
        document_info = self.storage.build_document_overview(self.chunks, self.hierarchy)

        return {
            "document_info": document_info,
            "chunks": None,
            "chunks_location": chunks_location,
            "hierarchy": self.hierarchy.__dict__ if self.hierarchy else {},
            "font_analysis": self.font_analysis.__dict__ if self.font_analysis else {},
            "stats": self.stats.__dict__ if self.stats else None,
            "total_chunks": len(self.chunks),
            "success": True,
        }

    def _generate_stats(self) -> ProcessingStats:
        """Generate processing statistics from the current chunks and font analysis."""
        header_count = len([c for c in self.chunks.values() if c.metadata.header_level <= 4])
        total_words = sum(len(c.content.split()) for c in self.chunks.values())
        font_sizes_count = len(self.font_analysis.font_distribution) if self.font_analysis else 0
        return ProcessingStats(
            total_chunks=len(self.chunks),
            total_headers=header_count,
            total_elements=total_words,
            font_sizes_analyzed=font_sizes_count,
        )
