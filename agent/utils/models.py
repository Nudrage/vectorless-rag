"""
Pydantic models for request/response and pipeline data validation.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# API Request / Response
# -----------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Validated chat message request."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User message content",
    )


class CreateSessionResponse(BaseModel):
    """Response after creating a session."""

    session_id: str = Field(..., description="Created session ID")


class MessageItem(BaseModel):
    """Single message in history."""

    type: str = Field(..., description="human or ai")
    content: str = Field(..., description="Message content")


class HistoryResponse(BaseModel):
    """Conversation history response."""

    messages: List[MessageItem] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Chat endpoint response."""

    response: str = Field(..., description="Agent response text")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Turn metadata")


# -----------------------------------------------------------------------------
# Retrieval Plan (Planner output)
# -----------------------------------------------------------------------------


UseCaseType = Literal[
    "listing_task",      # List ALL API endpoints or loggable events
    "schema_task",       # Complete field-level information
    "api_task",          # Specific API endpoint information
    "generic_qa_task",   # Information from specific section
    "miscellaneous_task" # Any other task
]


class RetrievalPlanModel(BaseModel):
    """Structured retrieval plan from Planner node."""

    use_case: UseCaseType = Field(
        default="generic_qa_task",
        description="Identified use case",
    )
    reasoning: str = Field(default="", description="Brief reasoning for the plan")


# -----------------------------------------------------------------------------
# Reasoning Trace
# -----------------------------------------------------------------------------


class ReasoningStep(BaseModel):
    """Single step in agent's reasoning process."""
    
    observation: str = Field(..., description="What you noticed from the previous step or tool result")
    reasoning: str = Field(..., description="Why this information is relevant and what it means")
    action: str = Field(..., description="What action you're taking next")
    expected_outcome: str = Field(..., description="What you expect to find or achieve")


# -----------------------------------------------------------------------------
# Planner Structured Outputs
# -----------------------------------------------------------------------------


class PlannerClarificationResponse(BaseModel):
    """When planner needs more info from user."""
    
    status: Literal["clarify"] = "clarify"
    question: str = Field(..., description="Clarifying question to ask user")
    reasoning: str = Field(default="", description="Why clarification is needed")


class PlannerProceedResponse(BaseModel):
    """When planner has enough info to proceed."""
    
    status: Literal["proceed"] = "proceed"
    section_number: str = Field(..., description="Section number from discover_documents (e.g., '5', '5.1')")
    retrieval_plan: RetrievalPlanModel = Field(..., description="Retrieval plan details")
    reasoning_trace: List[ReasoningStep] = Field(default_factory=list, description="Step-by-step reasoning process")


class PlannerResponse(BaseModel):
    """Union response from planner - either clarify or proceed."""
    
    response: PlannerClarificationResponse | PlannerProceedResponse = Field(
        ..., 
        discriminator="status",
        description="Planner decision"
    )


# -----------------------------------------------------------------------------
# Retriever Structured Output
# -----------------------------------------------------------------------------


class RetrieverClarificationResponse(BaseModel):
    """When retriever needs clarification from user."""
    
    status: Literal["clarify"] = "clarify"
    question: str = Field(..., description="Clarifying question to ask user")
    reasoning: str = Field(default="", description="Why clarification is needed")


class RetrieverProceedResponse(BaseModel):
    """When retriever has successfully retrieved chunks."""
    
    status: Literal["proceed"] = "proceed"
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    sections_covered: List[str] = Field(default_factory=list, description="Section numbers covered")
    reasoning_trace: List[ReasoningStep] = Field(default_factory=list, description="Step-by-step reasoning process")


class RetrieverResponse(BaseModel):
    """Union response from retriever - either clarify or proceed."""
    
    response: RetrieverClarificationResponse | RetrieverProceedResponse = Field(
        ...,
        discriminator="status",
        description="Retriever decision"
    )


# -----------------------------------------------------------------------------
# Transformer Structured Output
# -----------------------------------------------------------------------------


class TransformerResponse(BaseModel):
    """Final formatted response."""
    
    answer: str = Field(..., description="Final formatted answer")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


# -----------------------------------------------------------------------------
# Chunk (Retriever output)
# -----------------------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Chunk metadata (flexible for varying doc structures)."""

    model_config = {"extra": "allow"}


class ChunkModel(BaseModel):
    """Validated chunk from document tools."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    header: str = Field(default="", description="Section header text")
    content: str = Field(default="", description="Chunk text content")
    section_number: str = Field(default="", description="Section number")
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    relevance_score: Optional[float] = Field(default=None)

    model_config = {"extra": "allow"}


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def validate_message(value: str) -> str:
    """Validate and normalize user message."""
    msg = (value or "").strip()
    if not msg:
        raise ValueError("Message cannot be empty")
    if len(msg) > 10000:
        raise ValueError("Message exceeds maximum length of 10000 characters")
    return msg


def validate_session_id(value: str) -> str:
    """Validate session ID format."""
    if not value or not value.strip():
        raise ValueError("Session ID cannot be empty")
    if len(value) > 128:
        raise ValueError("Session ID too long")
    return value.strip()
