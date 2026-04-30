#!/usr/bin/env python3
"""
Chunker CLI: process PDF documents using RAG chunking.

Run from project root:
  python chunker/main.py document.pdf
  python -m chunker document.pdf
"""

import os
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add chunker directory to path so imports find utils, config, etc.
CHUNKER_DIR = Path(__file__).resolve().parent
if str(CHUNKER_DIR) not in sys.path:
    sys.path.insert(0, str(CHUNKER_DIR))

from config.settings import ChunkerSettings
from exceptions import (
    ChunkerError,
    ChunkerFileNotFoundError,
    InvalidFileError,
    ProcessingError,
)
from logging_config.setup import setup_logging
from utils import DocumentProcessor

try:
    from chunker import __version__
except ImportError:
    __version__ = "1.0.0"

logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60
CONTENT_PREVIEW_LENGTH = 150


def _parse_args() -> argparse.Namespace:
    """Build argument parser with version, verbose, output-dir, log-file, quiet."""
    parser = argparse.ArgumentParser(description="Process PDF documents using RAG chunking")

    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("pdf", nargs="?", help="Path to the PDF file to process")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v INFO, -vv DEBUG)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Override output directory for chunks")
    parser.add_argument("--log-file", type=Path, default=None, help="Write logs to this file (with rotation)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output and reduce logging")
    return parser.parse_args()


def process_pdf(pdf_path: str, output_dir: Optional[Path] = None, quiet: bool = False) -> Dict[str, Any]:
    """
    Process a PDF file using RAG chunking.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Optional override for output directory.
        quiet: If True, reduce progress and log output.

    Returns:
        Result dict with success, chunks_location, stats, total_chunks, etc.

    Raises:
        ChunkerFileNotFoundError: If the PDF file does not exist.
        InvalidFileError: If the file is not a PDF or is invalid.
        ProcessingError: If extraction, analysis, or chunking fails.

    Example:
        >>> result = process_pdf("report.pdf", output_dir=Path("out"))
        >>> print(result["total_chunks"])
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        logger.warning("PDF file not found: %s", pdf_path)
        raise ChunkerFileNotFoundError(f"PDF file not found: '{pdf_path}'")

    if pdf_file.suffix.lower() != ".pdf":
        logger.warning("File is not a PDF: %s", pdf_path)
        raise InvalidFileError(f"File is not a PDF: '{pdf_path}'")

    logger.info(SEPARATOR)
    logger.info("Processing: %s", pdf_path)
    logger.info(SEPARATOR)

    settings = ChunkerSettings()
    processor = DocumentProcessor(settings=settings)
    result = processor.process_document(pdf_file, output_dir=output_dir or settings.output_dir, quiet=quiet)

    if result.get("success"):
        _display_results(result, str(output_dir or settings.output_dir), pdf_file.stem)
        return result

    error_msg = result.get("error", "Unknown error")
    logger.error("Processing failed: %s", error_msg)
    raise ProcessingError(error_msg)


def _display_results(
    result: Dict[str, Any],
    output_dir: str,
    document_name: str,
) -> None:
    """Display processing results to the user."""
    logger.info(SEPARATOR)
    logger.info("Processing Complete!")
    logger.info(SEPARATOR)
    logger.info("Chunks created: %s", result["total_chunks"])
    output_folder = result.get("chunks_location") or f"{output_dir}/{document_name}/"
    logger.info("Output folder:  %s", output_folder)

    stats = result.get("stats") or {}
    if isinstance(stats, dict) and stats:
        logger.info("Statistics:")
        logger.info("  Headers detected: %s", stats.get("total_headers", 0))
        logger.info("  Font sizes found: %s", stats.get("font_sizes_analyzed", 0))

    chunks = result.get("chunks") or {}
    if chunks:
        logger.info("First 3 chunks:")
        for chunk_id, data in list(chunks.items())[:3]:
            header = data.get("header", "Untitled") if isinstance(data, dict) else getattr(data, "header", "Untitled")
            logger.info("  • %s: %s", chunk_id, header)


def main() -> None:
    """Main entry point for the Chunker CLI."""
    args = _parse_args()

    # if not args.pdf:
    #     logger.error("Missing required argument: pdf file path")
    #     sys.exit(2)

    settings = ChunkerSettings()
    log_level = "INFO"
    if args.verbose == 1:
        log_level = "INFO"
    elif args.verbose >= 2:
        log_level = "DEBUG"
    if args.quiet:
        log_level = "WARNING"

    setup_logging(level=log_level, log_file=args.log_file or settings.log_file, json_format=settings.log_json)

    try:
        directory_path = Path("data")
        for file in directory_path.rglob("*"):
            if file.is_file() and file.suffix.lower() == ".pdf":
                process_pdf(pdf_path=file, output_dir=args.output_dir, quiet=args.quiet)
    except ChunkerError as e:
        logger.exception("Chunker error: %s", e.message)
        sys.exit(e.exit_code)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
