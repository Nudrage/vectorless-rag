#!/usr/bin/env python3
"""
Custom exception hierarchy for the chunker package.

All exceptions inherit from ChunkerError and carry an exit_code for CLI usage.
Use ChunkerFileNotFoundError to avoid shadowing built-in FileNotFoundError.
"""


class ChunkerError(Exception):
    """Base exception for chunker package."""

    exit_code: int = 1

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class ChunkerFileNotFoundError(ChunkerError):
    """PDF file not found."""

    exit_code: int = 3


class InvalidFileError(ChunkerError):
    """File is not a valid PDF or has invalid format."""

    exit_code: int = 2


class ExtractionError(ChunkerError):
    """Error during PDF text extraction."""

    exit_code: int = 4


class ProcessingError(ChunkerError):
    """Error during document processing (chunking, analysis)."""

    exit_code: int = 4


class StorageError(ChunkerError):
    """Error saving chunks to disk."""

    exit_code: int = 5


class ConfigurationError(ChunkerError):
    """Invalid configuration."""

    exit_code: int = 6
