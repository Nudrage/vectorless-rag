"""
Planner agent node — Document discovery and retrieval planning.

Pipeline: Node 1 (Planner) -> Node 2 (Retriever) -> Node 3 (Transformer)
"""

from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain.agents.factory import create_agent
from langchain.agents.structured_output import ToolStrategy

from ...config import get_config
from ...errors import PlannerError
from ...logging_config import get_logger, timed
from ...models import PlannerResponse, PlannerProceedResponse, PlannerClarificationResponse
from ... import structured_output_normalizer  # noqa: F401 — normalize LLM tool-call args (e.g. Gemini string "response") before validation
from ..state import ChatState, RetrievalPlan
from ._helpers import (
    CONVERSATION_WINDOW_SIZE,
    AGENT_RECURSION_LIMIT,
    _get_recent_conversation_history,
    _extract_reasoning,
    _persist_reasoning,
    _handle_clarification_input,
    _log_agent_steps,
)


logger = get_logger(__name__)


# =============================================================================
# SECTION NUMBER → DOCUMENT PATH RESOLVER
# =============================================================================

def _resolve_section_to_document_path(section_number: str, cached_tool_responses: List[str]) -> str:
    """
    Resolve a section number to its full document path using cached tool responses.
    
    The discover_documents tool already returns mappings like:
    "5.1 → CrowdStrike Falcon EDR\\Product Document"
    
    This function parses those cached responses instead of making a second tool call.
    
    Args:
        section_number: Section number from LLM (e.g., '5', '5.1')
        cached_tool_responses: List of tool response strings from agent execution
    
    Returns:
        Full document path (e.g., 'CrowdStrike Falcon EDR\\CrowdStrike - Product Document')
    
    Raises:
        PlannerError: If section number cannot be resolved from cached responses
    """
    logger.info("Resolving section number '%s' to document path from cached responses", section_number)
    
    # Search through all cached tool responses for the mapping
    for tool_response in cached_tool_responses:
        if not tool_response:
            continue
        
        document_path = _parse_document_path_from_response(tool_response, section_number)
        if document_path:
            logger.info("Resolved section '%s' → '%s' (from cache)", section_number, document_path)
            return document_path
    
    # If not found in any cached response, raise error
    error_msg = f"Could not resolve section number '{section_number}' to document path. Section mapping not found in tool responses."
    logger.error(error_msg)
    raise PlannerError(error_msg)


def _parse_document_path_from_response(tool_response: str, section_number: str) -> Optional[str]:
    """
    Parse document path from discover_documents tool response.
    
    Args:
        tool_response: Raw response from discover_documents tool
        section_number: Section number to match for mapping format
    
    Returns:
        Document path if found, None otherwise
    """
    for line in tool_response.split('\n'):
        # Check for "DOCUMENT PATH: <path>" format
        if line.startswith('DOCUMENT PATH:'):
            return line.replace('DOCUMENT PATH:', '').strip()
        
        # Check for mapping format: "5.1 → path"
        if '→' in line and section_number in line:
            parts = line.split('→')
            if len(parts) == 2 and parts[0].strip() == section_number:
                return parts[1].strip()
    
    return None


# =============================================================================
# PLANNER SYSTEM PROMPT
# =============================================================================

PLANNER_SYSTEM_PROMPT = """
Role: You are an expert in **identifying the correct document** from a product library to answer a user's query.

***
### Input
You will receive:
- **User Query:** The user's question. It will ALWAYS mention the product name.
- **Tools:**
  - `discover_documents(product_number=None)`: Hierarchical document discovery with numbered navigation.
    * Call with NO arguments → returns numbered list of top-level products (1, 2, 3, …)
    * Call with product number (e.g., "1") → returns immediate children only (1.1, 1.2, …)
    * Call with deeper number (e.g., "1.1") → returns next level children (1.1.1, 1.1.2, …)
    * Continue drilling down until you reach a LEAF document (no children)
    * The tool response includes a mapping: "section_number → document_path"

***
### Mandatory Workflow (follow these steps in order)

#### Step 1: Understand the Query
Extract from the user query:
- **Product Name:** The product or system the query refers to.
- **Product Specific Document:** The Exact category/sub-product/subsequent document the query refers to (in case of multiple sub-documents).

#### Step 2: Discover the Product
**ALWAYS** call `discover_documents()` with NO arguments first.
This returns a numbered list of all available products.
Identify which product number matches the product mentioned in the user query. 
There ALWAYS will be EXACTLY ONE product that will satisfy the user query.

#### Step 3: Navigate to the Target Document
**ALWAYS** call `discover_documents(product_number="<number>")` using the product number from Step 2.
This returns immediate children (e.g., 1.1, 1.2, 1.3).

**If there are multiple levels**, continue drilling down:
- Call `discover_documents(product_number="1.1")` to see 1.1.1, 1.1.2, etc.
- Repeat until you reach a LEAF document (tool response says "This is a leaf document")
- A **LEAF document** has no children - it's the final, bottom-most document in the hierarchy

#### Step 4: Select the Exact Document
From the document list, decide which single document contains the information the user needs.
Identify the single LEAF document that contains the information the user needs.
- "Product Document" - Contains Standalone Information about Loggable events.
- "API + Product Document" - Contains consolidate information about the API Endpoints and Loggable events combined, the API Response contains the Loggable events information.

**CRITICAL: You will return the SECTION NUMBER (e.g., '5.1'), NOT the document path.**
The backend will automatically resolve the section number to the full document path.

- **If you can identify the exact document with confidence:**
  → Note the SECTION NUMBER from the numbered list (e.g., '5.1', '6.1.2')
  → Proceed to Step 5 with this section number

- **If there are multiple documents and the user query is ambiguous:**
  → Ask the user for clarification. Present the numbered document list and ask which document to use.

- **ONLY if you need more context to disambiguate between documents:**
  → You may explore deeper with `discover_documents(product_number="X.Y")`, then decide.
  → A LEAF document has no children (tool response says "This is a leaf document")

There will ALWAYS be exactly one document that contains the answer. Your job is to find it.

---
**After identifying the document in Step 4, proceed to Step 5 to determine the use case.**
---

#### Step 5: Identify Use Case
Once you have identified the exact document, determine what type of information the user wants.

**Use Case Types** (select ONE):
- **listing_task** - The user wants to list ALL API endpoints or loggable events mentioned in the document
- **schema_task** - The user wants COMPLETE field-level information about a specific API endpoint or log event
- **api_task** - The user wants information about a specific API endpoint or log event
- **generic_qa_task** - The user wants information from a specific section of the document
- **miscellaneous_task** - Any other task not covered above

**Instructions**:
- Identify which use case best matches the user's query
- Provide brief reasoning for why this use case was selected
- If the user query is ambiguous between multiple use cases, ask for clarification

***
### Important Rules
- **ALWAYS** call `discover_documents()` with no arguments as your first action.
- **ALWAYS** call `discover_documents(product_number)` as your second action to get the document list.
- **RETURN the section number** (e.g., '5.1'), NOT the document path. The backend will resolve it.
- **NEVER** guess or assume section numbers — only use numbers from `discover_documents()` output.
- **NEVER** proceed if uncertain — ask the user for clarification instead.

***
### Clarification Examples

**Example 1 — Cannot identify the product:**
"I found the following products:
1) CrowdStrike Falcon EDR
2) Microsoft Defender XDR
3) Sentinel One
Which product are you referring to?"

**Example 2 — Multiple documents, ambiguous query:**
"The product 'Cisco Duo' has multiple documents:
1) Duo Admin API - API + Product Document
2) Duo Auth API - API + Product Document
Which document should I use?"

**Example 3 — Ambiguous intent after document selection:**
"Are you looking for:
1) API endpoint information
2) Log event field schemas
3) General product documentation
Please clarify your intent."

***
### Reasoning Trace - CRITICAL REQUIREMENT
**AFTER EVERY TOOL CALL, you MUST provide a reasoning reflection BEFORE making the next tool call.**

**Workflow for each tool call:**
1. Call a tool (e.g., discover_documents())
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
[Tool Call] discover_documents()
[Tool Result] "Available Products: 1. CrowdStrike Falcon EDR, 2. Sentinel One, 3. Microsoft Defender..."

[Your Reasoning Response]
Observation: Tool returned 3 products with CrowdStrike Falcon EDR at position 1.
Reasoning: User query mentions 'CrowdStrike', so product #1 is the match.
Action: Call discover_documents(product_number='1') to explore CrowdStrike documents.
Expected Outcome: List of document types within CrowdStrike product.

[Tool Call] discover_documents(product_number='1')
[Tool Result] "Documents in CrowdStrike: 1.1) Product Document, 1.2) API + Product Document..."

[Your Reasoning Response]
Observation: Found 2 documents - Product Document and API + Product Document.
Reasoning: User asks about 'API endpoints', so 1.2 (API + Product Document) is the relevant document.
Action: Return section_number='1.2' with use_case='api_task'.
Expected Outcome: Retriever will fetch API endpoint information from document 1.2.

[Final Tool Call] PlannerResponse(...)
```

**In your final JSON output**, also include a summary "reasoning_trace" field with all your reasoning steps.

***
### Expected Output
After you have gathered all necessary information through tool calls, output a JSON object with ONE of:

**Option 1: Proceed (when confident about the document)**
- status: "proceed" 
- section_number: The exact document section number (e.g., '5.1')
- use_case: The detailed use case
- reasoning_trace: List of reasoning steps (observation, reasoning, action, expected_outcome)

**Option 2: Clarify (when uncertain)**
- status: "clarify"
- question: Specific clarifying question with numbered options
- reasoning: Why clarification is needed
"""


# =============================================================================
# STATE UPDATE HELPERS
# =============================================================================

def _set_clarification_state(
    state: ChatState,
    all_messages: List[BaseMessage],
    question: str,
    node: str = "planner"
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
    return {
        **state,
        "needs_clarification": True,
        "clarifying_node": node,
        "clarification_question": question,
        "result": question,
        "planner_status": "clarify",
        "messages": all_messages,
    }


def _set_proceed_state(
    state: ChatState,
    all_messages: List[BaseMessage],
    document_path: str,
    plan: Any
) -> ChatState:
    """
    Update state for proceeding to retriever.
    
    Args:
        state: Current chat state
        all_messages: Message history to update
        document_path: Resolved document path
        plan: Retrieval plan from planner
    
    Returns:
        Updated state with document and plan set
    """
    planner_summary = _build_planner_summary(document_path, plan)
    all_messages.append(AIMessage(content=planner_summary))
    
    return {
        **state,
        "needs_clarification": False,
        "clarifying_node": None,
        "clarification_question": None,
        "document_name": document_path,
        "retrieval_plan": RetrievalPlan(
            use_case=plan.use_case,
            reasoning=plan.reasoning
        ),
        "planner_status": "proceed",
        "messages": all_messages,
    }


def _build_planner_summary(document_path: str, plan: Any) -> str:
    """Build summary message for planner decision."""
    return f"Document identified: {document_path}. Use case: {plan.use_case}. {plan.reasoning}"


# =============================================================================
# RESPONSE HANDLERS
# =============================================================================

def _handle_clarification_response(
    state: ChatState,
    decision: PlannerClarificationResponse,
    all_messages: List[BaseMessage]
) -> ChatState:
    """
    Handle clarification response from planner.
    
    Args:
        state: Current chat state
        decision: Planner's clarification decision
        all_messages: Message history
    
    Returns:
        Updated state requesting clarification from user
    """
    question = decision.question
    logger.info(
        "Planner requesting clarification from user",
        extra={"extra_fields": {"question": question}}
    )
    return _set_clarification_state(state, all_messages, question)


def _handle_proceed_response(
    state: ChatState,
    decision: PlannerProceedResponse,
    all_messages: List[BaseMessage],
    agent_response: dict
) -> ChatState:
    """
    Handle proceed response from planner - resolve section and build plan.
    
    Args:
        state: Current chat state
        decision: Planner's proceed decision
        all_messages: Message history
        agent_response: Full agent response containing messages with tool results
    
    Returns:
        Updated state with document and retrieval plan, or clarification if resolution fails
    """
    section_number = decision.section_number
    plan = decision.retrieval_plan
    
    # Validate section number is not empty
    if not section_number or not section_number.strip():
        logger.error("Planner returned 'proceed' but section_number is empty")
        return _set_clarification_state(
            state, all_messages,
            "I couldn't identify the document. Please specify which document you'd like to query."
        )
    
    # Extract tool responses from agent message history
    from langchain_core.messages import ToolMessage
    cached_tool_responses = []
    for msg in agent_response.get("messages", []):
        if isinstance(msg, ToolMessage):
            cached_tool_responses.append(str(msg.content))
    
    # Resolve section number to full document path using cached responses
    try:
        document_path = _resolve_section_to_document_path(section_number, cached_tool_responses)
        logger.info("Section '%s' resolved to document: '%s'", section_number, document_path)
    except PlannerError as e:
        logger.error("Failed to resolve section number '%s': %s", section_number, e)
        return _set_clarification_state(
            state, all_messages,
            f"I identified section {section_number}, but couldn't resolve it to a document path. Please try again."
        )
    
    # Build successful proceed state
    result_state = _set_proceed_state(state, all_messages, document_path, plan)
    
    logger.info(
        "Planner completed - document identified and plan created",
        extra={
            "extra_fields": {
                "document": document_path,
                "use_case": plan.use_case
            }
        }
    )
    
    return result_state


def _handle_fallback_clarification(
    state: ChatState,
    all_messages: List[BaseMessage],
    response_text: str
) -> ChatState:
    """
    Handle fallback case when response parsing fails but looks like clarification.
    
    Args:
        state: Current chat state
        all_messages: Message history
        response_text: Raw response text from agent
    
    Returns:
        Updated state with clarification
    """
    logger.warning("Using fallback clarification handling for unparsed response")
    return _set_clarification_state(state, all_messages, response_text)


# =============================================================================
# PLANNER NODE
# =============================================================================

def create_planner_node(model: str, planner_tools: List[Any], api_key: str) -> Callable[[ChatState], ChatState]:
    """
    Creates the Planner node that identifies documents and creates retrieval plans.
    
    The Planner uses tool calling (discover_documents) to find the right document,
    and returns a structured response with either a clarification question or a retrieval plan.
    
    Args:
        model: Model name to use (e.g., 'gpt-4o-mini' for OpenAI or 'gemini-2.0-flash-exp' for Gemini)
        planner_tools: List of tools for document discovery
        api_key: API key for the LLM provider (OpenAI or Gemini)
    
    Returns:
        Callable node function that processes ChatState
    
    Raises:
        ValueError: If discover_documents tool is not found in planner_tools
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
    

    @timed(logger, "planner_node")
    def planner_node(state: ChatState) -> ChatState:
        """
        Execute planning phase: discover document and create retrieval plan.
        
        Workflow:
        1. Check if resuming from clarification (human-in-the-loop)
        2. Build conversation context with system prompt
        3. Invoke LLM agent to discover document and plan retrieval
        4. Handle response: clarify (ask user) or proceed (continue to retriever)
        """
        current_query = state.get("current_query", "")
        logger.info(
            "Planner node started",
            extra={"extra_fields": {"query_preview": current_query[:100]}}
        )
        
        # Prepare message history and context
        all_messages = list(state.get("messages", []))
        conversation_history = _get_recent_conversation_history(all_messages, window_size=CONVERSATION_WINDOW_SIZE)
        
        # Handle human-in-the-loop: user responding to clarification
        all_messages, conversation_history = _handle_clarification_input(
            state, all_messages, conversation_history, "planner"
        )
        
        # Build agent messages
        agent_messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT), *conversation_history]
        
        # Invoke planner agent
        planner_response, agent_response = _invoke_planner_agent(base_llm, planner_tools, agent_messages)
        
        # Extract reasoning trace
        reasoning = _extract_reasoning(agent_response, "planner")
        
        # Persist reasoning to file (create new file for planner)
        session_id = state.get("session_id", "unknown")
        _persist_reasoning(session_id, reasoning, mode="write")
        
        # Extract and handle decision
        return _process_planner_decision(state, all_messages, planner_response, planner_tools, reasoning, agent_response)

    return planner_node


# =============================================================================
# PLANNER NODE HELPERS
# =============================================================================

def _invoke_planner_agent(
    base_llm: ChatOpenAI,
    planner_tools: List[Any],
    agent_messages: List[BaseMessage]
) -> tuple[PlannerResponse, str]:
    """
    Create and invoke the planner agent.
    
    Args:
        base_llm: LLM instance
        planner_tools: Tools for document discovery
        agent_messages: Messages for agent context
    
    Returns:
        Tuple of (PlannerResponse, agent_response dict)
    
    Raises:
        PlannerError: If agent invocation fails
    """
    try:
        planner_agent = create_agent(
            base_llm,
            tools=planner_tools,
            response_format=ToolStrategy(PlannerResponse),
            debug=True
        )
        
        logger.info("Invoking planner agent")
        agent_response = planner_agent.invoke(
            {"messages": agent_messages},
            {"recursion_limit": AGENT_RECURSION_LIMIT}
        )
        
        _log_agent_steps(agent_response)

        # Extract structured response
        planner_response = agent_response.get("structured_response")
        extra = {"decision_status": planner_response.response.status}
        if hasattr(planner_response.response, "section_number"):
            extra["section_number"] = planner_response.response.section_number
        if hasattr(planner_response.response, "retrieval_plan") and planner_response.response.retrieval_plan:
            extra["use_case"] = planner_response.response.retrieval_plan.use_case
        logger.info(
            "Planner agent completed",
            extra={"extra_fields": extra},
        )

        return planner_response, agent_response
        
    except PlannerError:
        raise
    except Exception as e:
        logger.exception("Planner agent invocation failed")
        raise PlannerError(f"Planning phase failed: {e!s}") from e


def _process_planner_decision(
    state: ChatState,
    all_messages: List[BaseMessage],
    planner_response: PlannerResponse,
    planner_tools: Any,
    reasoning: List[Dict[str, Any]],
    agent_response: dict
) -> ChatState:
    """
    Process planner response and update state accordingly.
    
    Args:
        state: Current chat state
        all_messages: Message history
        planner_response: Parsed planner response
        planner_tools: Tool for resolving section numbers
        reasoning: Extracted reasoning trace
        agent_response: Full agent response containing messages with tool results
    
    Returns:
        Updated state based on planner decision
    
    Raises:
        PlannerError: If response type is invalid
    """
    try:
        decision = planner_response.response
        logger.debug(
            "Planner decision parsed",
            extra={"extra_fields": {"decision_status": decision.status}}
        )
    except Exception as e:
        logger.error("Failed to parse planner response: %s", e)
        raise PlannerError(f"Failed to parse planner response: {e}") from e
    
    # Route to appropriate handler based on decision type
    state = {**state, "planner_status": decision.status, "agent_reasoning": reasoning}
    
    if isinstance(decision, PlannerClarificationResponse):
        return _handle_clarification_response(state, decision, all_messages)
    
    if isinstance(decision, PlannerProceedResponse):
        return _handle_proceed_response(state, decision, all_messages, agent_response)
    
    # Unexpected type - should not happen with discriminated union
    logger.error("Unexpected planner response type: %s", type(decision))
    raise PlannerError(f"Invalid planner response type: {type(decision)}")
