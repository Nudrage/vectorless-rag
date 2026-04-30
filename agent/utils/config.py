"""
Centralized configuration for the agent framework.

Uses Pydantic BaseModel for type validation.
All settings are developer-configured with hardcoded defaults.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM and API configuration with multi-provider support (OpenAI and Gemini)."""

    provider: str = Field(default="", description="LLM provider ('openai' or 'gemini')")
    model: str = Field(default="", description="LLM model name")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries for LLM API calls")
    retry_initial_delay: float = Field(default=1.0, ge=0.1, description="Initial retry delay in seconds")
    retry_max_delay: float = Field(default=300.0, ge=1.0, description="Max retry delay in seconds")
    api_key: str = Field(default="", description="API key from environment")
    
    def __init__(self, **data):
        """Initialize LLMConfig with automatic provider detection from environment variables."""
        super().__init__(**data)
        
        if not self.provider or not self.model or not self.api_key:
            openai_key = os.getenv("OPENAI_API_KEY", "")
            google_key = os.getenv("GOOGLE_API_KEY", "")
            
            if openai_key and google_key:
                raise ValueError(
                    "Both OPENAI_API_KEY and GOOGLE_API_KEY found in environment. "
                    "Please use only one LLM provider at a time."
                )
            elif openai_key:
                self.provider = "openai"
                self.api_key = openai_key
                if not self.model:
                    self.model = "gpt-4o-mini"
            elif google_key:
                self.provider = "gemini"
                self.api_key = google_key
                if not self.model:
                    self.model = "gemini-3.1-flash-lite-preview"
            else:
                raise ValueError(
                    "No LLM API key found in environment. "
                    "Please set either OPENAI_API_KEY or GOOGLE_API_KEY."
                )


class PipelineConfig(BaseModel):
    """Pipeline and graph execution configuration."""

    max_iterations: int = Field(default=5, ge=1, le=200, description="Max tool-call iterations per node")
    timeout_seconds: int = Field(default=300, ge=30, le=3600, description="Pipeline timeout per turn")
    chunks_dir: str = Field(default="output", description="Directory containing document chunks")


class LoggingConfig(BaseModel):
    """Structured logging configuration."""

    level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    json_format: bool = Field(default=False, description="Use JSON format for console log output")
    include_correlation_id: bool = Field(default=True, description="Include correlation ID in logs")
    log_to_file: bool = Field(default=True, description="Write logs to a file")
    log_file_path: Optional[str] = Field(default=None, description="Log file path (default: logs/agent.log)")
    file_json_format: bool = Field(default=False, description="Use JSON format for log file output")


class AgentConfig(BaseModel):
    """Root configuration aggregating all sub-configs."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# Singleton for application-wide use
_config: Optional[AgentConfig] = None


def get_config() -> AgentConfig:
    """Return the application configuration (singleton)."""
    global _config
    if _config is None:
        _config = AgentConfig()
    return _config


def set_config(config: AgentConfig) -> None:
    """Set the application configuration (for testing)."""
    global _config
    _config = config
