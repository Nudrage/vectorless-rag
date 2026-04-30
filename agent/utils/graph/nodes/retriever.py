"""
Retriever agent node — Chunk retrieval based on the planner's retrieval plan.

Pipeline: Node 1 (Planner) -> Node 2 (Retriever) -> Node 3 (Transformer)
"""

import json
from typing import Any, Callable, Dict, List, Optional

from functools import wraps

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents.factory import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import create_model

from ...config import get_config
from ...errors import RetrieverError
from ...logging_config import get_logger, timed
from ...models import RetrieverResponse, RetrieverClarificationResponse, RetrieverProceedResponse
from ... import structured_output_normalizer  # noqa: F401 — normalize LLM tool-call args (e.g. Gemini string "response") before validation
from ..state import ChatState
from ._helpers import (
    CONVERSATION_WINDOW_SIZE,
    AGENT_RECURSION_LIMIT,
    _get_recent_conversation_history,
    _extract_reasoning,
    _persist_reasoning,
    _handle_clarification_input,
    _log_agent_steps,
)
from ...tools.read_outline import read_outline
from ...tools.get_chunks_by_section import get_chunks_by_section
from ...tools.search_chunks import search_chunks
from ...tools.regex_search_chunks import regex_search_chunks


logger = get_logger(__name__)


# =============================================================================
# RETRIEVER SYSTEM PROMPT
# =============================================================================

RETRIEVER_SYSTEM_PROMPT = """
You are a Chunk Retriever for cybersecurity documentation. Your job is to:
1. Explore the document structure using the outline
2. Identify the exact sections containing the information
3. Retrieve relevant chunks based on the use case

CONTEXT PROVIDED:
- Document path: Already set (you don't need to specify it in tool calls)
- Use case: The type of information to retrieve
- User query: The original question

WORKFLOW:

Step 1: EXPLORE DOCUMENT STRUCTURE

**Terminology**:
- **LEAF section**: A section with no subsections (no children in the outline tree).
  Example: If outline shows "1.2.3 (0 sub levels present)" or just "1.2.3" with no indicator,
  it's a LEAF section. If it shows "1.2 (5 sub levels present)", it's NOT a leaf.

The outline file contains all the headers present in the document in the form of a tree structure. Thus the sub headers are detailed information about the main headers.
Do note that these headers act as the metadata about the information that is stored in their respective chunks. 
These are all documents about a Cyber Security product, thus the headings will be related to the respective domains of those products.

**Exploration Process**:
- Call read_outline() with NO arguments to see Level 1 headers
- Each header shows "(N sub levels present)" if it has children
- Call read_outline(expand_section="X") to expand sections that match query keywords
- **Expansion Strategy**:
  * Expand if header name contains query-relevant keywords
  * Continue expanding until "(0 sub levels present)" or no sub-level indicator
  * A LEAF section = section with no children (no sub-level indicator)
  * Stop expanding branches that don't match the query
- Identify the exact LEAF section(s) containing the answer

Step 2: DOCUMENT UNDERSTANDING 
- "API + Product Document" - Consolidate information about the API Endpoints and Loggable Events, 
  - The headers will contain information about the API and not about the exact Endpoint.
  - The information will be about API's that product support and the Request and Responses of that API 
  - the Response Format of the API endpoints contains information about the Schema of Loggable Events.
- "Product Document" - Standalone information about the Loggable Events.
  - The headers will be related to the loggable events that the product performs

Step 3: IDENTIFY SECTIONS BASED ON USE CASE
For "listing_task" (listing API Endpoints and Loggable Events):
- Here the User want to know about the List of ALL the API ENDPOINTS or LOGGABLE EVENTS provided by the product within the document. 
- Here there won't be the section finding task, but the listing task and that should be followed by the human intervention, you list it. Human review it and provide the further requirements. 
Case 1: When the items of the list is present in the headers (outline of the document)
  - The headers of the document will contain ALL the items required for listing, 
  - thus the requirement from the agent would be to identify the headers from the outline.
Case 2: When the items of the list is present within a single particular chunk of the document. 
  - There will be a heading whose content will include the information about the list of API Endpoints or Loggable Events. 
  - thus the requirement from the agent would be to identify that particular chunk, read that chunk and then list down the endpoints within the content of that chunk.

For "schema_task" (Schema information about the specific loggable events):
- Here the User want to know about ALL THE fields, and all information about those fields present in a specific loggable events provided by the product within the document. 
Case 1: When ALL the information is scattered within multiple chunks of the document. 
  - In this case the fields are strcutured in multiple sections as per classes or sub-container of the loggable events.
  - The outline file will contain the heading of sub-container.
  - The requirement from the agent would be to identify ALL THE chunks or sub-container that would contain the information. 
Case 2: When ALL the information is present within a single particular chunk of the document. 
  - The outline file will contain the heading of Loggable Events.
  - The requirement from the agent would be to identify the chunk the information about the fields for that specified event.
Case 3: When the document contains both the API and Product Document
  - In such cases the Response format of the API will contain the schema for that particular event. 
  - In such cases the API will be present, followed by the request format, followed by response format. 
  - The requirement will be to identify the particular single chunk response format realted to the requested event present in the heading of the outline file.

For "api_task" (API information):
Case 1: When ALL the information is present within a single particular chunk of the document.
  - Look for the heading that matches the user query and the respective chunk will contain the complete information, request and response format. 
Case 2: When the information is segregated in multiple chunks.
  - Look for the heading that matches the user query. There will be a separate heading for request and response format. 
  - If the user asks for a specific section (request or response) then return the respective chunk, else return both the chunks.

For "generic_qa_task" and "miscellaneous_task" (specific information):
- Identify relevant section from outline which seems to contain the information about the user query. 
- Retrieve chunks with get_chunks_by_section(section_number)
- Make use of tools to find the required information in cases of unability to find using header.
- Ask the user by providing options of the headers that which heading is the one which will contain the information about the user query. 

Step 4: ASK FOR CLARIFICATION WHEN NEEDED
- Multiple sections could contain the answer
- Header names are ambiguous
- Pattern matching returns too many/few results
- ALWAYS provide numbered options to the user

TOOLS AVAILABLE (document path is auto-injected):
- read_outline(expand_section=None): Read document outline. Start with no args for Level 1, then expand by section number
- get_chunks_by_section(section_number): Get chunks for exact section (e.g., "1.2.3")
- search_chunks(query, max_results=10): Keyword search across document
- regex_search_chunks(pattern): Regex search, returns section numbers and headers

Note: You DON'T need to specify document_name - it's automatically set to the document identified by the Planner.

REASONING TRACE - CRITICAL REQUIREMENT:
**AFTER EVERY TOOL CALL, you MUST provide a reasoning reflection BEFORE making the next tool call.**

**Workflow for each tool call:**
1. Call a tool (e.g., read_outline())
2. Receive the tool result
3. **IMMEDIATELY provide a text response** with your reasoning in this format:
   ```
   Observation: [What you noticed from the tool result]
   Reasoning: [Why this information is relevant to the user's query]
   Action: [What you will do next]
   Expected Outcome: [What you expect to find or achieve]
   ```
4. Then make your next tool call (or final response)

**Example conversation flow:**
```
[Tool Call] read_outline()
[Tool Result] "1. Admin API (3 sub levels present), 2. Auth API (2 sub levels present)..."

[Your Reasoning Response]
Observation: Outline shows Admin API with 3 sub-levels and Auth API with 2 sub-levels.
Reasoning: User asked about Admin API endpoints, so section 1 is relevant and needs expansion.
Action: Call read_outline(expand_section='1') to see Admin API subsections.
Expected Outcome: Will see the structure of Admin API sections to identify endpoint listings.

[Tool Call] read_outline(expand_section='1')
[Tool Result] "1.1 Users API, 1.2 Groups API, 1.3 Logs API..."

[Your Reasoning Response]
Observation: Admin API has subsections for Users, Groups, and Logs APIs.
Reasoning: These are the API categories. User wants all endpoints, so I need all these sections.
Action: Call get_chunks_by_section for sections 1.1, 1.2, and 1.3.
Expected Outcome: Retrieve chunks containing all Admin API endpoint information.

[Tool Call] get_chunks_by_section(section_number='1.1')
...
```

Track your reasoning throughout the retrieval process. **In your final JSON output**, also include a summary "reasoning_trace" field with all your reasoning steps.

OUTPUT FORMAT (required):
After retrieving, output a JSON object with ONE of:

**Option 1: Proceed (when chunks retrieved successfully)**
- status: "proceed"
- chunks_retrieved: Number of chunks retrieved
- sections_covered: List of section numbers (e.g., ["1.5.4", "1.5.4.1"])
- reasoning_trace: List of reasoning steps (observation, reasoning, action, expected_outcome)

**Option 2: Clarify (when uncertain)**
- status: "clarify"
- question: Specific clarifying question with numbered options
- reasoning: Why clarification is needed

IMPORTANT:
- Always start by exploring the outline
- **For listing_task**: 
  * If list items are in headers: Call get_chunks_by_section() for those sections
  * If list items are in content: Call get_chunks_by_section() to retrieve the chunk containing the list
  * The chunks will contain the list data - you don't manually extract it
- **For schema_task and other tasks**: Call get_chunks_by_section() to retrieve full chunks
- Ask for clarification with numbered options when uncertain
- DO NOT work on assumptions - always validate with tools before retrieving"""


# =============================================================================
# RETRIEVER NODE
# =============================================================================


# =============================================================================
# CONTEXTUAL TOOL BINDING
# =============================================================================

def _bind_document_path(original_tool, document_path: str) -> StructuredTool:
    """
    Bind document_path to a tool, removing it from the agent-facing signature.
    
    The original tool's description is preserved so the agent sees full documentation.
    
    Args:
        original_tool: LangChain tool with document_name as first parameter
        document_path: Document path to inject automatically
    
    Returns:
        New tool with document_name pre-bound
    """
    original_func = original_tool.func
    
    @wraps(original_func)
    def bound_func(**kwargs):
        kwargs["document_name"] = document_path
        return original_func(**kwargs)
    
    # Remove document_name from the schema for the agent
    original_schema = original_tool.args_schema
    if original_schema:
        # Filter out document_name from schema fields
        filtered_fields = {
            k: v for k, v in original_schema.__fields__.items() 
            if k != "document_name"
        }
        # Create new schema without document_name
        new_schema = create_model(
            f"{original_tool.name}_contextual_schema",
            **{k: (v.annotation, v.default) for k, v in filtered_fields.items()}
        )
    else:
        new_schema = None
    
    return StructuredTool.from_function(
        func=bound_func,
        name=original_tool.name,
        description=original_tool.description,
        args_schema=new_schema,
    )


def _create_contextual_tools(document_path: str) -> List:
    """
    Create tools with document_path pre-bound for the retriever agent.
    
    Args:
        document_path: Document path to inject into all tool calls
    
    Returns:
        List of tools with document_name automatically injected
    """
    return [
        _bind_document_path(read_outline, document_path),
        _bind_document_path(get_chunks_by_section, document_path),
        _bind_document_path(search_chunks, document_path),
        _bind_document_path(regex_search_chunks, document_path),
    ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_tool_content(content: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Parse tool content into list of chunks.
    
    Args:
        content: Raw content from ToolMessage
    
    Returns:
        List of chunk dictionaries, or None if parsing fails
    """
    try:
        if isinstance(content, str):
            if content.startswith('[') or content.startswith('{'):
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and 'chunks' in data:
                    return data['chunks']
        elif isinstance(content, list):
            return content
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse tool content as JSON: %s", e)
    except Exception as e:
        logger.warning("Unexpected error parsing tool content: %s", e)
    return None


def _extract_chunks_from_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Extract chunks from tool message responses.
    
    Args:
        messages: List of messages from agent response
    
    Returns:
        List of extracted chunk dictionaries
    """
    chunks = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            extracted = _parse_tool_content(msg.content)
            if extracted:
                chunks.extend(extracted)
    return chunks


def _extract_sections_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    """Extract unique section numbers from chunks."""
    return list(set(
        chunk.get('section_number', '')
        for chunk in chunks
        if chunk.get('section_number')
    ))


def _build_retrieval_summary(chunks_count: int, sections: List[str]) -> str:
    """Build summary message for retrieval completion."""
    sections_preview = ', '.join(sections[:10])
    suffix = '...' if len(sections) > 10 else ''
    return f"Retrieved {chunks_count} chunks from sections: {sections_preview}{suffix}. Ready for transformation."


# =============================================================================
# STATE UPDATE HELPERS
# =============================================================================

def _set_clarification_state(
    state: ChatState,
    all_messages: List[BaseMessage],
    question: str,
    node: str = "retriever"
) -> ChatState:
    """
    Update state for clarification request.
    
    Args:
        state: Current chat state
        all_messages: Message history to update
        question: Clarification question to ask user
        node: Node requesting clarification
    
    Returns:
        Updated state with clarification flags set
    """
    all_messages.append(AIMessage(content=question))
    logger.info(
        "Retriever requesting clarification",
        extra={"extra_fields": {"question": question[:200]}}
    )
    return {
        **state,
        "needs_clarification": True,
        "clarifying_node": node,
        "clarification_question": question,
        "result": question,
        "messages": all_messages,
    }


def _handle_clarification_response(
    state: ChatState,
    decision: RetrieverClarificationResponse,
    all_messages: List[BaseMessage]
) -> ChatState:
    """
    Handle clarification response from retriever.

    Args:
        state: Current chat state
        decision: Retriever's clarification decision
        all_messages: Message history

    Returns:
        Updated state requesting clarification from user
    """
    return _set_clarification_state(state, all_messages, decision.question, "retriever")


def _set_success_state(
    state: ChatState,
    all_messages: List[BaseMessage],
    chunks: List[Dict[str, Any]],
    sections: List[str],
    reasoning: List[Dict[str, Any]]
) -> ChatState:
    """
    Update state for successful retrieval.
    
    Args:
        state: Current chat state
        all_messages: Message history to update
        chunks: Retrieved chunks
        sections: Sections covered
        reasoning: Extracted reasoning trace
    
    Returns:
        Updated state with retrieval results
    """
    summary = _build_retrieval_summary(len(chunks), sections)
    all_messages.append(AIMessage(content=summary))
    
    logger.info(
        "Retriever completed",
        extra={
            "extra_fields": {
                "chunks_retrieved": len(chunks),
                "sections_covered": len(sections),
                "sections": sections[:5]
            }
        }
    )
    
    return {
        **state,
        "chunks": chunks,
        "needs_clarification": False,
        "clarifying_node": None,
        "clarification_question": None,
        "messages": all_messages,
        "agent_reasoning": reasoning,
    }


# =============================================================================
# AGENT INVOCATION HELPERS
# =============================================================================

def _build_retrieval_prompt(current_query: str, document_path: str, retrieval_plan: dict) -> str:
    """Build the retrieval prompt for the agent."""
    use_case = retrieval_plan.get("use_case", "generic_qa")
    target_sections = retrieval_plan.get('target_sections', [])
    search_queries = retrieval_plan.get('search_queries', [])
    
    return f"""User Query: {current_query}

Document: {document_path}
Use Case: {use_case}
Search Queries: {search_queries}

Please explore the document structure and retrieve the relevant chunks based on the use case and retrieval plan.
Follow the workflow in the system prompt."""


def _invoke_retriever_agent(
    base_llm: ChatOpenAI,
    tools: List[Callable],
    messages: List[BaseMessage]
) -> tuple[List[BaseMessage], RetrieverResponse, dict]:
    """
    Create and invoke the retriever agent.
    
    Args:
        base_llm: LLM instance
        tools: Context-aware tools for document retrieval
        messages: Messages for agent context
    
    Returns:
        Tuple of (response_messages, retriever_response, agent_response)
    
    Raises:
        RetrieverError: If agent invocation fails
    """
    try:
        retrieval_agent = create_agent(
            base_llm,
            tools=tools,
            response_format=ToolStrategy(RetrieverResponse),
            debug=True
        )
        
        logger.info("Invoking retriever agent")
        agent_response = retrieval_agent.invoke(
            {"messages": messages},
            {"recursion_limit": AGENT_RECURSION_LIMIT}
        )
        
        _log_agent_steps(agent_response)
        
        response_messages = agent_response.get("messages", [])
        retriever_response = agent_response.get("structured_response")
        extra = {"decision_status": retriever_response.response.status}
        if hasattr(retriever_response.response, "chunks_retrieved"):
            extra["chunks_count"] = retriever_response.response.chunks_retrieved
        if hasattr(retriever_response.response, "sections_covered"):
            extra["sections_covered"] = retriever_response.response.sections_covered
        logger.info(
            "Retriever agent completed",
            extra={"extra_fields": extra},
        )

        return response_messages, retriever_response, agent_response
        
    except RetrieverError:
        raise
    except Exception as e:
        logger.exception("Retriever agent invocation failed")
        raise RetrieverError(f"Retrieval phase failed: {e!s}") from e


def _process_retrieval_results(
    state: ChatState,
    all_messages: List[BaseMessage],
    retriever_response: RetrieverResponse,
    response_messages: List[BaseMessage],
    reasoning: List[Dict[str, Any]]
) -> ChatState:
    """
    Process retriever response - either clarify or proceed.
    
    Args:
        state: Current chat state
        all_messages: Message history
        retriever_response: Structured response from retriever
        response_messages: Messages from agent response
        reasoning: Extracted reasoning trace
    
    Returns:
        Updated state based on retriever decision
    
    Raises:
        RetrieverError: If response type is invalid
    """
    try:
        decision = retriever_response.response
        logger.debug(
            "Retriever decision parsed",
            extra={"extra_fields": {"decision_status": decision.status}}
        )
    except Exception as e:
        logger.error("Failed to parse retriever response: %s", e)
        raise RetrieverError(f"Failed to parse retriever response: {e}") from e
    
    # Route to appropriate handler based on decision type
    if isinstance(decision, RetrieverClarificationResponse):
        return _handle_clarification_response(state, decision, all_messages)
    
    if isinstance(decision, RetrieverProceedResponse):
        # Extract chunks from tool messages
        retrieved_chunks = _extract_chunks_from_messages(response_messages)
        
        # Validate chunks were retrieved
        if not retrieved_chunks:
            logger.warning(
                "No chunks extracted from tool responses",
                extra={"extra_fields": {"message_count": len(response_messages)}}
            )
        
        # Get sections from structured response or extract from chunks
        sections_covered = decision.sections_covered or _extract_sections_from_chunks(retrieved_chunks)
        
        return _set_success_state(state, all_messages, retrieved_chunks, sections_covered, reasoning)
    
    # Unexpected type - should not happen with discriminated union
    logger.error("Unexpected retriever response type: %s", type(decision))
    raise RetrieverError(f"Invalid retriever response type: {type(decision)}")


# =============================================================================
# RETRIEVER NODE
# =============================================================================

def create_retriever_node(model: str, api_key: str) -> Callable[[ChatState], ChatState]:
    """
    Creates the Retriever node that explores document structure and retrieves chunks.
    
    The Retriever uses context-aware tools that automatically inject the document path
    from the Planner's decision, so the LLM only needs to specify section numbers or queries.
    
    Args:
        model: Model name to use (e.g., 'gpt-4o-mini' for OpenAI or 'gemini-2.0-flash-exp' for Gemini)
        api_key: API key for the LLM provider (OpenAI or Gemini)
    
    Returns:
        Callable node function that processes ChatState
    """
    from ...llm_factory import get_llm_instance
    
    config = get_config()
    
    # Initialize LLM for agent using factory (supports OpenAI and Gemini)
    base_llm = get_llm_instance(
        provider=config.llm.provider,
        model=model,
        api_key=api_key,
        temperature=config.llm.temperature,
        verbose=False
    )

    @timed(logger, "retriever_node")
    def retriever_node(state: ChatState) -> ChatState:
        """
        Execute retrieval phase: explore document structure and retrieve chunks.
        
        Workflow:
        1. Extract document path and use case from state
        2. Create context-aware tools with injected document path
        3. Build agent with contextual tools
        4. Invoke agent to explore outline and retrieve chunks
        5. Handle clarification or return results
        """
        # Extract state values
        document_path = state.get("document_name", "")
        retrieval_plan = state.get("retrieval_plan", {}) or {}
        current_query = state.get("current_query", "")
        use_case = retrieval_plan.get("use_case", "generic_qa")
        
        # Validate required state
        if not document_path:
            logger.error("No document path provided by Planner")
            raise RetrieverError("Document path is missing from state")
        
        logger.info(
            "Retriever node started",
            extra={"extra_fields": {"document_path": document_path, "use_case": use_case}}
        )
        
        # Prepare message history and context
        all_messages = list(state.get("messages", []))
        conversation_history = _get_recent_conversation_history(all_messages, window_size=CONVERSATION_WINDOW_SIZE)
        
        # Handle human-in-the-loop: user responding to clarification
        all_messages, conversation_history = _handle_clarification_input(
            state, all_messages, conversation_history, "retriever"
        )
        
        # Create context-aware tools with document path injected
        contextual_tools = _create_contextual_tools(document_path)
        
        # Build agent messages
        retrieval_prompt = _build_retrieval_prompt(current_query, document_path, retrieval_plan)
        agent_messages = [
            SystemMessage(content=RETRIEVER_SYSTEM_PROMPT),
            *conversation_history,
            HumanMessage(content=retrieval_prompt)
        ]
        
        # Invoke retriever agent
        response_messages, retriever_response, agent_response = _invoke_retriever_agent(
            base_llm, contextual_tools, agent_messages
        )
        
        # Extract reasoning trace
        reasoning = _extract_reasoning(agent_response, "retriever")
        
        # Persist reasoning to file (append to planner's file)
        session_id = state.get("session_id", "unknown")
        _persist_reasoning(session_id, reasoning, mode="append")
        
        # Process results and update state
        return _process_retrieval_results(
            state, all_messages, retriever_response, response_messages, reasoning
        )

    return retriever_node
