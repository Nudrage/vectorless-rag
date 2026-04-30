#!/usr/bin/env python3
"""
Centralized constants for the chunker package.

All hardcoded values used across extractor, analyzer, chunker, storage, and core
are defined here for easier configuration and consistency.
"""

# --- Core / processor (header detection) ---
DEFAULT_MIN_HEADER_FREQUENCY = 0.05
DEFAULT_MAX_HEADER_FREQUENCY = 0.4
DEFAULT_FONT_SIZE_TOLERANCE = 0.5

# --- Analyzer (font analysis) ---
MAX_COMMON_FREQ = 0.2  # Headers are rare; skip font sizes more common than this

# --- Chunker (header detection and merging) ---
MAX_HEADER_TEXT_LENGTH = 150
MIN_FONT_SIZE_THRESHOLD = 8.0
LEFT_MARGIN_THRESHOLD = 100
MAX_VERTICAL_GAP_FOR_MERGE = 30.0  # Max vertical gap in pixels to merge headers
FONT_SIZE_TOLERANCE_FOR_MERGE = 0.5
MIN_HEADER_TEXT_LENGTH = 2

# --- Storage (file naming and format) ---
TIMESTAMP_FORMAT = "%Y%m%d_%H%M"
CHUNKS_FILE_PREFIX = "chunks_"
HIERARCHY_FILE_PREFIX = "hierarchy_"
FONTS_FILE_PREFIX = "fonts_"
JSON_EXTENSION = ".json"

# --- Logging (rotation) ---
LOG_ROTATION_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
