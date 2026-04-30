"""
Document agent framework for RAG systems using LangGraph.

3-Node Pipeline Architecture:
    Node 1 (Planner) -> Node 2 (Retriever) -> Node 3 (Transformer)

Main Entry Points:
    - ChatAgenticFramework: High-level multi-turn chat interface
    - create_rag_graph: Low-level graph factory for custom integrations

Production utilities:
    - config: Centralized configuration (get_config, AgentConfig)
    - errors: Exception hierarchy, retry, circuit breaker
    - logging_config: Structured logging, correlation IDs
    - models: Pydantic validation models
"""

from .core import ChatAgenticFramework
from .graph import ChatState, RetrievalPlan, create_rag_graph
from .config import get_config, AgentConfig, LLMConfig, PipelineConfig, LoggingConfig
from .errors import (
    AgentError,
    ConfigurationError,
    ValidationError,
    PipelineError,
    PlannerError,
    RetrieverError,
    TransformerError,
    LLMError,
    RateLimitError,
    LLMAPIError,
    DocumentError,
    DocumentNotFoundError,
    ChunkNotFoundError,
    RetryExhaustedError,
    CircuitBreakerOpenError,
    retry_with_backoff,
    CircuitBreaker,
    circuit_breaker,
)
from .logging_config import (
    get_logger,
    get_correlation_id,
    set_correlation_id,
    new_correlation_id,
    configure_logging,
    timed,
)
from .models import (
    ChatRequest,
    ChatResponse,
    CreateSessionResponse,
    MessageItem,
    HistoryResponse,
    RetrievalPlanModel,
    ChunkModel,
    validate_message,
    validate_session_id,
)

__all__ = [
    # High-level interface
    'ChatAgenticFramework',
    # Graph
    'ChatState',
    'RetrievalPlan',
    'create_rag_graph',
    # Config
    'get_config',
    'AgentConfig',
    'LLMConfig',
    'PipelineConfig',
    'LoggingConfig',
    # Errors
    'AgentError',
    'ConfigurationError',
    'ValidationError',
    'PipelineError',
    'PlannerError',
    'RetrieverError',
    'TransformerError',
    'LLMError',
    'RateLimitError',
    'LLMAPIError',
    'DocumentError',
    'DocumentNotFoundError',
    'ChunkNotFoundError',
    'RetryExhaustedError',
    'CircuitBreakerOpenError',
    'retry_with_backoff',
    'CircuitBreaker',
    'circuit_breaker',
    # Logging
    'get_logger',
    'get_correlation_id',
    'set_correlation_id',
    'new_correlation_id',
    'configure_logging',
    'timed',
    # Models
    'ChatRequest',
    'ChatResponse',
    'CreateSessionResponse',
    'MessageItem',
    'HistoryResponse',
    'RetrievalPlanModel',
    'ChunkModel',
    'validate_message',
    'validate_session_id',
]
