#!/usr/bin/env python3
"""
Document text extraction functionality.

Handles PDF text extraction with font and positioning information,
providing the foundation for subsequent font analysis and chunking.
Each line is assigned a dominant font (by character count across spans);
font properties include size and name (weight/style when implemented).

Example:
    >>> from utils.extractor import DocumentExtractor
    >>> extractor = DocumentExtractor()
    >>> elements = extractor.extract_text_elements("document.pdf")
    >>> for element in elements:
    ...     print(f"{element.text[:50]}... (size: {element.font.size})")
"""

import fitz
import logging
from pathlib import Path
from typing import Dict, List, Union

from tqdm import tqdm

from .models import FontInfo, TextElement

logger = logging.getLogger(__name__)

try:
    from exceptions import InvalidFileError, ExtractionError
except ImportError:
    InvalidFileError = Exception  # type: ignore[misc, assignment]
    ExtractionError = Exception  # type: ignore[misc, assignment]


class DocumentExtractor:
    """
    Handles PDF text extraction with enhanced information.

    Extracts text elements from PDF documents along with their dominant font
    (by character count per line) and positioning, enabling font-based analysis.
    """

    def __init__(self) -> None:
        pass

    def extract_text_elements(self, pdf_path: Union[str, Path]) -> List[TextElement]:
        """
        Extract text elements from a PDF document with font and positioning information.

        Uses PyMuPDF (fitz) to parse the PDF at the line level. For each line,
        the dominant font is chosen by character count. Each line becomes a
        TextElement with text, font properties, page number, bbox, and line number.

        Args:
            pdf_path: Path to the PDF file to process.

        Returns:
            List of TextElement objects, one per text line.

        Raises:
            InvalidFileError: If the file is corrupted or not a valid PDF.
            ExtractionError: If text extraction fails.
        """
        path_str = str(pdf_path)
        try:
            document = fitz.open(path_str)
        except fitz.FileDataError as e:
            logger.warning("Corrupted or invalid PDF: %s", path_str)
            raise InvalidFileError(f"Corrupted or invalid PDF: {path_str}") from e
        except OSError as e:
            logger.warning("Could not open file: %s - %s", path_str, e)
            raise ExtractionError(f"Could not open file: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error opening PDF: %s", path_str)
            raise ExtractionError(f"Failed to open PDF: {e}") from e

        try:
            with document:
                total_pages = len(document)
                text_elements: List[TextElement] = []

                logger.info("Starting extraction from '%s' (%s pages)", path_str, total_pages)

                with tqdm(
                    total=total_pages,
                    desc="Extracting pages",
                    unit="page",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] {postfix}',
                ) as pbar:
                    for page_number in range(total_pages):
                        page = document[page_number]
                        page_text_blocks = page.get_text("dict", sort=True)
                        logger.debug("Page %s: %s blocks", page_number + 1, len(page_text_blocks.get("blocks", [])))

                        line_counter = 0
                        for block in page_text_blocks["blocks"]:
                            if "lines" not in block:
                                continue
                            for line in block["lines"]:
                                line_text = ""
                                line_span_fonts: List[Dict] = []
                                line_bounding_box = line["bbox"]

                                for span in line["spans"]:
                                    font_size = span["size"]
                                    font_name = span["font"]
                                    text = span["text"]
                                    line_text += text
                                    line_span_fonts.append({
                                        "size": font_size,
                                        "name": font_name,
                                        "text": text,
                                    })

                                if line_text.strip():
                                    size_to_length: Dict[float, int] = {}
                                    size_to_name: Dict[float, str] = {}
                                    for span in line_span_fonts:
                                        s, n, t = span["size"], span["name"], span["text"]
                                        size_to_length[s] = size_to_length.get(s, 0) + len(t)
                                        if s not in size_to_name:
                                            size_to_name[s] = n
                                    dominant_size = max(size_to_length, key=size_to_length.get) if size_to_length else 0
                                    dominant_name = size_to_name.get(dominant_size, "Unknown")
                                    font_info = FontInfo(size=dominant_size, name=dominant_name)

                                    element = TextElement(
                                        text=line_text.strip(),
                                        font=font_info,
                                        page=page_number + 1,
                                        bbox=line_bounding_box,
                                        line_number=line_counter,
                                    )
                                    text_elements.append(element)
                                    line_counter += 1

                        pbar.set_postfix_str(f"{len(text_elements)} elements")
                        pbar.update(1)

                logger.info("Extraction complete: %s text elements from %s page(s)", len(text_elements), total_pages)
                return text_elements
        except (InvalidFileError, ExtractionError):
            raise
        except Exception as e:
            logger.exception("Extraction failed for %s", path_str)
            raise ExtractionError(f"Extraction failed: {e}") from e
