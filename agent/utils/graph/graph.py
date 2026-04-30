"""
LangGraph definition for the 3-node pipeline framework.

Pipeline Architecture:
    Node 1 (Planner) -> Node 2 (Retriever) -> Node 3 (Transformer) -> END

Each node has focused responsibilities:
    - Planner: Document discovery and retrieval planning
    - Retriever: Chunk fetching based on plan
    - Transformer: Content formatting into final output
"""

from langgraph.graph import StateGraph, END

from ..logging_config import get_logger
from .state import ChatState
from .nodes.planner import create_planner_node
from .nodes.retriever import create_retriever_node
from .nodes.transformer import create_transformer_node

logger = get_logger(__name__)


def route_after_planner(state: ChatState) -> str:
    """Route based on planner decision (only 2 outcomes: clarify or proceed)."""
    if state.get("needs_clarification"):
        logger.info("Planner needs clarification, routing to end")
        return "end"

    planner_status = state.get("planner_status")
    if planner_status == "proceed":
        document_name = state.get("document_name")
        if document_name and document_name.strip():
            logger.info("Routing to retriever",
                        extra={"extra_fields": {"document": document_name}})
            return "retriever"
        logger.error("Planner status is 'proceed' but no valid document_name found")
        return "end"

    logger.info("Routing to end - planner_status: %s", planner_status)
    return "end"


def route_after_retriever(state: ChatState) -> str:
    """Route based on retriever clarification needs and validation."""
    if state.get("needs_clarification"):
        logger.info("Retriever needs clarification, routing to end")
        return "end"

    chunks = state.get("chunks", [])
    if chunks and len(chunks) > 0:
        logger.info("Routing to transformer",
                    extra={"extra_fields": {"chunks_count": len(chunks)}})
        return "transformer"
    logger.error("Retriever completed but no chunks were retrieved")
    return "end"


def route_after_transformer(state: ChatState) -> str:
    """Route based on transformer clarification needs."""
    if state.get("needs_clarification"):
        logger.info("Transformer needs clarification, routing to end")
        return "end"
    logger.info("Routing to end")
    return "end"


def create_rag_graph(model: str, chunks_dir: str, api_key: str) -> StateGraph:
    """
    Creates and compiles the 3-node LangGraph pipeline.

    Pipeline:
        Planner -> Retriever -> Transformer -> END

    Args:
        model: Model name to use (e.g., 'gpt-4o-mini' for OpenAI or 'gemini-2.0-flash-exp' for Gemini)
        chunks_dir: Chunks directory path
        api_key: API key for the LLM provider (OpenAI or Gemini)

    Returns:
        Compiled StateGraph ready for execution
    """
    from ..tools import initialize_tools, get_planner_tools

    # Initialize tools with chunks directory
    initialize_tools(chunks_dir)

    # Get tools for planner (retriever builds contextual tools internally)
    planner_tools = get_planner_tools()
    
    # Create workflow
    workflow = StateGraph(ChatState)
    
    # Create node functions
    planner_node = create_planner_node(model, planner_tools, api_key)
    retriever_node = create_retriever_node(model, api_key)
    transformer_node = create_transformer_node(model, api_key)
    
    # Add nodes to graph
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("transformer", transformer_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Conditional routing from planner
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {"retriever": "retriever", "end": END}
    )

    # Conditional routing from retriever
    workflow.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {"transformer": "transformer", "end": END}
    )

    # Conditional routing from transformer
    workflow.add_conditional_edges(
        "transformer",
        route_after_transformer,
        {"end": END}
    )
    
    # Compile graph
    app = workflow.compile()
    
    logger.info("LangGraph compiled: Planner -> [Conditional] -> Retriever -> Transformer pipeline")
    return app
