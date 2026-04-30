"""
LangGraph components for the 3-node pipeline framework.

Pipeline Architecture:
    Node 1 (Planner) -> Node 2 (Retriever) -> Node 3 (Transformer)

Exports:
    - ChatState: State schema for the pipeline
    - RetrievalPlan: Typed dict for retrieval planning
    - create_rag_graph: Factory function to create the compiled graph
    - Node creators for individual node instantiation
    - System prompts for each node
"""

from .state import ChatState, RetrievalPlan
from .graph import create_rag_graph
from .nodes.planner import create_planner_node, PLANNER_SYSTEM_PROMPT
from .nodes.retriever import create_retriever_node, RETRIEVER_SYSTEM_PROMPT
from .nodes.transformer import create_transformer_node, TRANSFORMER_SYSTEM_PROMPT

__all__ = [
    # State
    'ChatState',
    'RetrievalPlan',
    # Graph
    'create_rag_graph',
    # Nodes
    'create_planner_node',
    'create_retriever_node',
    'create_transformer_node',
    # Prompts
    'PLANNER_SYSTEM_PROMPT',
    'RETRIEVER_SYSTEM_PROMPT',
    'TRANSFORMER_SYSTEM_PROMPT',
]
