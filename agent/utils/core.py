"""
Main framework orchestrator for the 3-node pipeline RAG system using LangGraph.

Pipeline Architecture:
    Node 1 (Planner) -> Node 2 (Retriever) -> Node 3 (Transformer)
"""

import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage

from .config import get_config
from .errors import AgentError, PipelineError, ValidationError
from .graph import ChatState, create_rag_graph
from .logging_config import get_logger, new_correlation_id
from .models import validate_message, validate_session_id

logger = get_logger(__name__)


def _empty_session_state(session_id: str) -> ChatState:
    """Build initial ChatState for a new session."""
    return {
        "session_id": session_id,
        "messages": [],
        "current_query": "",
        "document_name": None,
        "retrieval_plan": None,
        "planner_status": None,
        "clarification_question": None,
        "chunks": [],
        "transformation_instructions": None,
        "result": None,
        "needs_clarification": False,
        "clarifying_node": None,
        "user_clarification_response": None,
        "metadata": {},
        "agent_reasoning": [],
    }


class ChatAgenticFramework:
    """
    Multi-turn chat framework with session management and 3-node LangGraph pipeline.

    Pipeline:
        Planner -> Retriever -> Transformer
    """

    def __init__(
        self, api_key: Optional[str] = None, model: Optional[str] = None, chunks_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the framework with LLM provider (OpenAI or Gemini) and LangGraph pipeline.

        Args:
            api_key: LLM API key (defaults to config / OPENAI_API_KEY or GOOGLE_API_KEY)
            model: LLM model name (defaults to config, e.g., 'gpt-4o-mini' for OpenAI or 'gemini-2.0-flash-exp' for Gemini)
            chunks_dir: Directory containing document chunks (defaults to config)
        """
        config = get_config()
        self.chunks_dir = chunks_dir or config.pipeline.chunks_dir
        self.llm_model = model or config.llm.model
        self.api_key = api_key or config.llm.api_key

        self.graph = create_rag_graph(
            model=self.llm_model, chunks_dir=self.chunks_dir, api_key=self.api_key
        )
        self._sessions: Dict[str, ChatState] = {}
        self._sessions_lock = threading.Lock()

        logger.info(
            "ChatAgentic framework initialized",
            extra={"extra_fields": {"model": self.llm_model, "chunks_dir": self.chunks_dir}}
        )

    def create_session(self) -> str:
        """Create a new chat session. Returns session_id."""
        session_id = str(uuid.uuid4())
        with self._sessions_lock:
            self._sessions[session_id] = _empty_session_state(session_id)
        logger.info("Created session", extra={"extra_fields": {"session_id": session_id}})
        return session_id

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Return conversation history for a session (list of messages)."""
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if not state:
            return []
        return list(state.get("messages", []))

    def has_session(self, session_id: str) -> bool:
        """Return True if the session exists."""
        with self._sessions_lock:
            return session_id in self._sessions

    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Return metadata from the last turn for a session."""
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if not state:
            return {}
        return dict(state.get("metadata", {}))

    def get_session_state(self, session_id: str) -> Optional[ChatState]:
        """Return the full session state for debugging/inspection."""
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        """Remove a session from memory."""
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    @contextmanager
    def session_context(self, session_id: Optional[str] = None) -> Iterator[str]:
        """
        Context manager for automatic session lifecycle management.

        Args:
            session_id: Optional existing session ID. If None, creates a new session.

        Yields:
            session_id: The session ID to use for chat operations.

        Example:
            with framework.session_context() as sid:
                response = framework.chat(sid, "Hello")
        """
        sid = session_id if session_id and self.has_session(session_id) else self.create_session()
        try:
            yield sid
        finally:
            if session_id is None:
                self.delete_session(sid)

    def _build_clarification_state(self, session_state: ChatState, user_message: str) -> ChatState:
        """Build state when resuming from a clarification request."""
        messages = list(session_state.get("messages", []))
        messages.append(HumanMessage(content=user_message))
        return {
            **session_state,
            "user_clarification_response": user_message,
            "current_query": user_message,
            "messages": messages,
        }

    def _build_fresh_state(self, session_state: ChatState, session_id: str, user_message: str) -> ChatState:
        """Build initial state for a new turn."""
        messages = list(session_state.get("messages", []))
        messages.append(HumanMessage(content=user_message))
        return {
            "session_id": session_id,
            "messages": messages,
            "current_query": user_message,
            "document_name": session_state.get("document_name"),
            "retrieval_plan": None,
            "planner_status": None,
            "clarification_question": None,
            "chunks": [],
            "transformation_instructions": None,
            "result": None,
            "needs_clarification": False,
            "clarifying_node": None,
            "user_clarification_response": None,
            "metadata": {},
            "agent_reasoning": [],
        }

    def _extract_response(self, final_state: ChatState) -> str:
        """Extract the appropriate response from final state."""
        if final_state.get("needs_clarification"):
            return final_state.get("clarification_question") or "Could you provide more details?"
        return final_state.get("result") or "No response generated."

    def chat(self, session_id: str, user_message: str) -> str:
        """
        Process one user message in the given session and return the agent response.

        Pipeline flow:
            1. Planner: Discovers document and creates retrieval plan
            2. Retriever: Fetches relevant chunks based on plan
            3. Transformer: Formats chunks into final response

        Args:
            session_id: Session identifier
            user_message: User's query message

        Returns:
            Agent response string

        Raises:
            ValidationError: If session_id or user_message is invalid
        """
        try:
            session_id = validate_session_id(session_id)
            user_message = validate_message(user_message)
        except ValueError as e:
            raise ValidationError(str(e)) from e

        with self._sessions_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = _empty_session_state(session_id)
            session_state = self._sessions[session_id]

        # Build initial state without mutating session_state
        if session_state.get("needs_clarification"):
            logger.info(
                "Resuming from clarification",
                extra={"extra_fields": {"clarifying_node": session_state.get("clarifying_node")}}
            )
            initial_state = self._build_clarification_state(session_state, user_message)
        else:
            initial_state = self._build_fresh_state(session_state, session_id, user_message)

        correlation_id = new_correlation_id()
        query_preview = user_message[:80] + "..." if len(user_message) > 80 else user_message
        logger.info(
            "Chat turn started",
            extra={
                "extra_fields": {
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "query_preview": query_preview,
                }
            },
        )

        start_time = time.perf_counter()
        try:
            final_state = self.graph.invoke(initial_state)
        except AgentError:
            raise
        except Exception as e:
            logger.exception(
                "Chat turn failed",
                extra={"extra_fields": {"correlation_id": correlation_id, "session_id": session_id}}
            )
            raise PipelineError(f"Error processing message: {e!s}") from e
        finally:
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Chat turn completed",
                extra={
                    "extra_fields": {
                        "correlation_id": correlation_id,
                        "session_id": session_id,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                },
            )

        with self._sessions_lock:
            self._sessions[session_id] = final_state

        # Log turn outcome (single line for sequence visibility)
        if final_state.get("needs_clarification"):
            clarifying_node = final_state.get("clarifying_node")
            question = final_state.get("clarification_question", "")
            logger.info(
                "%s needs clarification",
                clarifying_node,
                extra={"extra_fields": {"question_preview": question[:100] if question else ""}},
            )
            logger.info(
                "Turn outcome",
                extra={
                    "extra_fields": {
                        "outcome": "clarification",
                        "clarifying_node": clarifying_node,
                        "question_preview": question[:100] if question else "",
                    }
                },
            )
        else:
            logger.info(
                "Turn outcome",
                extra={"extra_fields": {"outcome": "success"}},
            )

        return self._extract_response(final_state)
