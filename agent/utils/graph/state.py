"""
LangGraph state schema for Chat Agentic framework.

This module defines the state that flows through the 3-node pipeline:
    Node 1 (Planner) -> Node 2 (Retriever) -> Node 3 (Transformer)

Conversation History Management:
    - Each node maintains the full message history in state["messages"]
    - Nodes use a sliding window of the last 3 conversation turns to prevent context overflow
    - All agent interactions (clarifications, decisions, results) are appended to history
    - This ensures multi-turn conversations maintain context while staying within token limits
"""

from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage


class RetrievalPlan(TypedDict, total=False):
    """
    Structured plan output from the Planner node.
    
    Fields:
        use_case: Identified use case (e.g., "listing_task", "schema_task", "api_task", "generic_qa_task", "miscellaneous_task")
        reasoning: Brief explanation of why this use case was chosen
    """
    use_case: str
    reasoning: str


class ChatState(TypedDict, total=False):
    """
    State that flows through the LangGraph 3-node pipeline.

    === Session Fields ===
    session_id: Unique session identifier
    messages: Full conversation history (includes tool calls and results)
              Each node appends its interactions to this list.
              Nodes use a 3-turn sliding window when invoking LLMs to prevent context overflow.
    current_query: Latest user message

    === Node 1 (Planner) Output ===
    document_name: Identified document folder name (e.g. from discover_documents)
                   MUST be validated as non-empty before proceeding to retriever
    retrieval_plan: Structured plan with sections to retrieve and use case
    planner_status: Routing decision ("proceed" | "clarify") - only 2 valid values
    clarification_question: Question to ask user when status is "clarify"

    === Node 2 (Retriever) Output ===
    chunks: Retrieved chunks for current turn (list of dicts with section_number, header, content)
    transformation_instructions: Instructions for how to transform chunks into output

    === Node 3 (Transformer) Output ===
    result: Agent response text for current turn
            In testing mode, this contains a summary of retrieved chunks

    === Human-in-the-Loop Fields ===
    needs_clarification: Flag indicating if any node needs human input
    clarifying_node: Which node is asking ("planner", "retriever", "transformer")
    user_clarification_response: User's answer to clarification question
                                 Cleared after being processed by the respective node

    === Metadata ===
    metadata: Processing metadata (chunks_used, sections_used, use_case, testing_mode, etc.)
    """
    # Session fields
    session_id: str
    messages: List[BaseMessage]
    current_query: str
    
    # Planner output
    document_name: Optional[str]
    retrieval_plan: Optional[RetrievalPlan]
    planner_status: Optional[str]
    clarification_question: Optional[str]
    
    # Retriever output
    chunks: List[Dict[str, Any]]
    transformation_instructions: Optional[str]
    
    # Transformer output
    result: Optional[str]
    
    # Human-in-the-loop fields
    needs_clarification: bool
    clarifying_node: Optional[str]
    user_clarification_response: Optional[str]
    
    # Metadata
    metadata: Dict[str, Any]
    
    # Agent reasoning trace
    agent_reasoning: List[Dict[str, Any]]
