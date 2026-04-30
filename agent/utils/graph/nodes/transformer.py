"""
Transformer agent node — Content transformation and output formatting.

Pipeline: Node 1 (Planner) -> Node 2 (Retriever) -> Node 3 (Transformer)
"""

from typing import Any, Callable, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ...config import get_config
from ...errors import TransformerError
from ...logging_config import get_logger, timed
from ..state import ChatState
from ._helpers import _RETRYABLE_EXCEPTIONS, _get_ai_message_content, _get_recent_conversation_history


logger = get_logger(__name__)


# =============================================================================
# TRANSFORMER SYSTEM PROMPT
# =============================================================================

TRANSFORMER_SYSTEM_PROMPT = """You are an Output Transformer for cybersecurity documentation. Your job is to:
1. Transform retrieved chunk content into the requested output format
2. Format the output based on the use case

CONTEXT PROVIDED:
- chunks: List of retrieved chunks with header, content, section_number
- current_query: The original user question
- use_case: The type of task (listing_task, schema_task, api_task, generic_qa_task, miscellaneous_task)

OUTPUT GUIDELINES BY USE CASE:

FOR schema_task (Field-level schema information):
- Extract field names from chunk content (usually in tables or lists)
- For each field: name, type (inferred), description, enum values (if any), examples
- Output valid JSON Schema format:
```json
{
  "type": "object",
  "properties": {
    "fieldName": {
      "type": "string",
      "description": "Field description from docs",
      "enum": ["value1", "value2"],
      "examples": ["example1"]
    }
  }
}
```

FOR listing_task (List API endpoints or loggable events):
- If listing API endpoints:
  * Format: `METHOD /path/to/endpoint - Brief description`
  * Group by resource/category if applicable
- If listing loggable events:
  * Format: `- EventName: Brief description`
  * Group by category if applicable

FOR api_task (API endpoint information):
- Provide complete API endpoint details
- Include request format, response format, parameters
- Use clear markdown formatting with sections

FOR generic_qa_task and miscellaneous_task:
- Provide a clear, direct answer
- Cite specific sections/chunks when relevant
- Use bullet points for lists

IMPORTANT:
- Be comprehensive - include all relevant information from chunks
- Be accurate - only include information present in the chunks
- Be formatted - use appropriate markdown formatting
- If chunks are insufficient to fully answer, state what's missing"""


# =============================================================================
# TRANSFORMER NODE
# =============================================================================


def create_transformer_node(model: str, api_key: str) -> Callable[[ChatState], ChatState]:
    """
    Creates the Transformer node in TESTING MODE.
    
    This node is currently disabled for testing chunk retrieval.
    It only logs the retrieved chunks without performing transformation.
    The full transformer agent will be implemented in the future.
    
    Args:
        model: Model name to use (e.g., 'gpt-4o-mini' for OpenAI or 'gemini-2.0-flash-exp' for Gemini)
        api_key: API key for the LLM provider (OpenAI or Gemini)
    
    Returns:
        Callable node function that processes ChatState
    """

    config = get_config()

    @timed(logger, "transformer_node")
    def transformer_node(state: ChatState) -> ChatState:
        """
        TESTING MODE: Log retrieved chunks without transformation.
        
        This node validates that chunks are correctly retrieved by the Retriever.
        No agent invocation or transformation is performed.
        """
        chunks = state.get("chunks", [])
        current_query = state.get("current_query", "")
        retrieval_plan = state.get("retrieval_plan", {}) or {}
        use_case = retrieval_plan.get("use_case", "generic_qa")

        # Get conversation history
        all_messages = list(state.get("messages", []))

        sections = list(set(c.get("section_number", "") for c in chunks if c.get("section_number")))
        chunks_summary = [
            {
                "section_number": c.get("section_number", "N/A"),
                "header": c.get("header", "N/A"),
                "chunk_id": c.get("chunk_id", "N/A"),
                "content_preview": (c.get("content") or "")[:200],
            }
            for c in chunks
        ]

        if not chunks:
            logger.warning("No chunks retrieved by Retriever node")
            result_message = "No chunks were retrieved. Please check the retrieval process."
        else:
            for i, chunk in enumerate(chunks, 1):
                logger.debug(
                    "Chunk %d: section=%s header=%s chunk_id=%s content_preview=%s... length=%d",
                    i,
                    chunk.get("section_number", "N/A"),
                    chunk.get("header", "N/A"),
                    chunk.get("chunk_id", "N/A"),
                    (chunk.get("content") or "")[:200],
                    len(chunk.get("content", "")),
                )

            result_message = f"""Retrieved {len(chunks)} chunks successfully from {len(sections)} sections.

Sections covered: {', '.join(sections)}

Transformer is in TESTING MODE. Check logs for detailed chunk information.

To enable transformation, the transformer agent will be implemented in the future."""

        logger.info(
            "Transformer node completed (testing mode)",
            extra={
                "extra_fields": {
                    "user_query_preview": current_query[:100] if current_query else "",
                    "use_case": use_case,
                    "chunks_count": len(chunks),
                    "sections": sections,
                    "chunks": chunks_summary,
                }
            },
        )

        # Update state
        state["needs_clarification"] = False
        state["clarifying_node"] = None
        state["clarification_question"] = None
        state["result"] = result_message

        # Set metadata
        state["metadata"] = {
            "total_chunks": len(chunks),
            "chunks_used": [c.get("chunk_id") for c in chunks if c.get("chunk_id")],
            "sections_used": list(set(c.get("section_number") for c in chunks if c.get("section_number"))),
            "use_case": use_case,
            "testing_mode": True,
        }

        # Update message history
        from langchain_core.messages import AIMessage
        all_messages.append(AIMessage(content=result_message))
        state["messages"] = all_messages

        return state

    return transformer_node
