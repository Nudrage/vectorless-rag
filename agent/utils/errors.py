"""
Custom exception hierarchy and resilience utilities for the agent framework.

- Exception hierarchy for precise error handling (used throughout the pipeline)
- Retry decorator (retry_with_backoff): reserved for future use on LLM/tool calls
- Circuit breaker: reserved for future use on external API resilience
"""

import functools
import time
from enum import Enum
from typing import Any, Callable, Optional, Set, TypeVar, Union

from .logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# -----------------------------------------------------------------------------
# Exception Hierarchy
# -----------------------------------------------------------------------------


class AgentError(Exception):
    """Base exception for all agent framework errors."""

    def __init__(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.message = message
        self.details = kwargs
        super().__init__(message, *args)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | {self.details}"
        return self.message


class ConfigurationError(AgentError):
    """Raised when configuration is invalid or missing."""

    pass


class ValidationError(AgentError):
    """Raised when input or output validation fails."""

    pass


class PipelineError(AgentError):
    """Base for pipeline node errors."""

    pass


class PlannerError(PipelineError):
    """Raised when the Planner node fails."""

    pass


class RetrieverError(PipelineError):
    """Raised when the Retriever node fails."""

    pass


class TransformerError(PipelineError):
    """Raised when the Transformer node fails."""

    pass


class LLMError(AgentError):
    """Base for LLM/API errors."""

    pass


class RateLimitError(LLMError):
    """Raised when API rate limit is exceeded."""

    pass


class LLMAPIError(LLMError):
    """Raised when LLM API call fails (network, auth, server error)."""

    pass


class DocumentError(AgentError):
    """Base for document/tool errors."""

    pass


class DocumentNotFoundError(DocumentError):
    """Raised when a requested document does not exist."""

    pass


class ChunkNotFoundError(DocumentError):
    """Raised when no chunks match the request."""

    pass


# -----------------------------------------------------------------------------
# Retry with Exponential Backoff
# -----------------------------------------------------------------------------


class RetryExhaustedError(AgentError):
    """Raised when all retries have been exhausted."""

    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    reraise_type: type = RetryExhaustedError,
) -> Callable[[F], F]:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exception types to retry
        reraise_type: Exception type to raise when retries exhausted
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.warning(
                            "Retries exhausted for %s after %d attempts",
                            func.__name__,
                            max_retries + 1,
                            exc_info=True,
                            extra={
                                "extra_fields": {
                                    "function": func.__name__,
                                    "attempts": max_retries + 1,
                                    "last_error": str(e),
                                }
                            },
                        )
                        raise reraise_type(
                            f"Retries exhausted after {max_retries + 1} attempts: {e!s}"
                        ) from e
                    logger.info(
                        "Attempt %d/%d failed for %s, retrying in %.2fs: %s",
                        attempt + 1,
                        max_retries + 1,
                        func.__name__,
                        delay,
                        e,
                        extra={
                            "extra_fields": {
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries + 1,
                                "delay_seconds": round(delay, 2),
                                "error": str(e),
                            }
                        },
                    )
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
            raise reraise_type(
                f"Retries exhausted: {last_exception!s}"
            ) from last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


# -----------------------------------------------------------------------------
# Circuit Breaker
# -----------------------------------------------------------------------------


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Test if service recovered


class CircuitBreakerOpenError(AgentError):
    """Raised when circuit breaker is open and call is rejected."""

    pass


class CircuitBreaker:
    """
    Circuit breaker for external API calls.

    - CLOSED: Calls pass through. On failure threshold, transition to OPEN.
    - OPEN: Calls fail immediately. After timeout, transition to HALF_OPEN.
    - HALF_OPEN: One call allowed. Success -> CLOSED, Failure -> OPEN.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "default",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._failure_count = 0
        return self._state

    def _record_success(self) -> None:
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute func through the circuit breaker.

        Raises:
            CircuitBreakerOpenError: When circuit is OPEN
        """
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. Try again later."
            )
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def __call__(self, func: F) -> F:
        """Use as decorator: @circuit_breaker."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)

        return wrapper  # type: ignore[return-value]


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator that wraps a function with a circuit breaker.

    Args:
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds before trying again (half-open)
        name: Circuit name (defaults to function name)
    """

    def decorator(func: F) -> F:
        cb = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name=name or func.__name__,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return cb.call(func, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
