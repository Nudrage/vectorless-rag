#!/usr/bin/env python3
"""
Pydantic-based settings for the chunker package.

Settings can be overridden via environment variables with the CHUNKER_ prefix
or via a .env file in the working directory.
"""

from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkerSettings(BaseSettings):
    """Application settings with environment and .env support."""

    model_config = SettingsConfigDict(env_prefix="CHUNKER_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    log_json: bool = False

    # Processing (header/font analysis)
    min_header_frequency: float = 0.05
    max_header_frequency: float = 0.4
    font_size_tolerance: float = 0.5
    max_common_freq: float = 0.2

    # Storage
    output_dir: Path = Path("output")
    source_data_dir: Path = Path("data")

    # Limits
    max_file_size_mb: int = 100

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("output_dir", "source_data_dir", "log_file", mode="before")
    @classmethod
    def coerce_path(cls, v: object) -> object:
        if v is None:
            return v
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            return Path(v)
        return v
