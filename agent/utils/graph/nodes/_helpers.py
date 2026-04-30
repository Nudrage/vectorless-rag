"""
Shared utilities used across agent node implementations.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ...logging_config import get_correlation_id, get_logger

logger = get_logger(__name__)


# Shared node constants (planner and retriever)
CONVERSATION_WINDOW_SIZE = 3
AGENT_RECURSION_LIMIT = 25

# Transient errors that are worth retrying (e.g. LLM API)
_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


def _extract_json_from_response(content: str) -> Dict[str, Any]:
    """Extract JSON object from LLM response text (used by retriever/transformer)."""
    if not content:
        return {}
    
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, content)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        pass
    
    return {}


def _get_ai_message_content(msg: AIMessage) -> str:
    """Extract text content from AIMessage."""
    content = msg.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif hasattr(part, 'text'):
                text_parts.append(part.text)
            elif isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
        return " ".join(text_parts)
    return str(content) if content else ""


def _find_final_response(messages: List) -> str:
    """Find the final AI response from message list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, 'tool_calls', None)
            if not tool_calls or (isinstance(tool_calls, list) and len(tool_calls) == 0):
                return _get_ai_message_content(msg)
    return ""


def _get_recent_conversation_history(messages: List[BaseMessage], window_size: int = 3) -> List[BaseMessage]:
    """
    Get the last N conversation turns (user-assistant pairs) to prevent context overflow.
    
    This function filters the message history to keep only the most recent conversation turns,
    excluding system messages. A "turn" consists of a user message and the corresponding
    assistant response (which may include tool calls and tool messages).
    
    Args:
        messages: Full message history from state
        window_size: Number of recent conversation turns to keep (default: 3)
    
    Returns:
        List of recent messages representing the last window_size conversation turns
    
    Example:
        If window_size=3 and there are 10 turns, only the last 3 turns (6-10) are returned.
    """
    if not messages:
        return []
    
    # Group messages into conversation turns
    # A turn starts with HumanMessage and includes all following messages until next HumanMessage
    turns = []
    current_turn = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Start a new turn
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            # Add to current turn (AIMessage, ToolMessage, etc.)
            if current_turn:  # Only add if we've started a turn
                current_turn.append(msg)
    
    # Add the last turn if it exists
    if current_turn:
        turns.append(current_turn)
    
    # Keep only the last window_size turns
    recent_turns = turns[-window_size:] if len(turns) > window_size else turns
    
    # Flatten the turns back into a message list
    recent_messages = []
    for turn in recent_turns:
        recent_messages.extend(turn)
    
    return recent_messages


def _handle_clarification_input(
    state: Dict[str, Any],
    all_messages: List[BaseMessage],
    conversation_history: List[BaseMessage],
    clarifying_node: str,
) -> tuple:
    """
    Handle user's clarification response if resuming from clarification.

    Args:
        state: Current chat state
        all_messages: Full message history
        conversation_history: Recent conversation window
        clarifying_node: Node name that requested clarification ("planner" or "retriever")

    Returns:
        Tuple of (updated all_messages, updated conversation_history)
    """
    user_response = state.get("user_clarification_response")
    is_resuming = user_response and state.get("clarifying_node") == clarifying_node

    if is_resuming:
        logger.info(
            "Resuming from clarification - user provided response",
            extra={"extra_fields": {"response_preview": user_response[:100]}}
        )
        clarification_msg = HumanMessage(content=user_response)
        conversation_history = [*conversation_history, clarification_msg]
        all_messages = [*all_messages, clarification_msg]

    return all_messages, conversation_history


def _log_agent_steps(agent_response: dict) -> None:
    """Log agent execution steps: one INFO summary; per-step detail at DEBUG."""
    messages = agent_response.get("messages", [])
    tool_call_count = sum(
        len(getattr(msg, "tool_calls", []) or [])
        for msg in messages
    )
    logger.info(
        "Agent returned %d messages (%d tool calls)",
        len(messages),
        tool_call_count,
        extra={
            "extra_fields": {
                "message_count": len(messages),
                "tool_call_count": tool_call_count,
            }
        },
    )
    for i, msg in enumerate(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_names = [tc.get("name") for tc in msg.tool_calls]
            logger.debug("Step %d: Tool calls - %s", i, tool_names)
        logger.debug("Step %d: %s - %s", i, type(msg).__name__, str(msg)[:200])


def _extract_reasoning(agent_response: dict, node_name: str) -> List[Dict[str, Any]]:
    """
    Extract reasoning from agent response including structured trace and tool calls.
    
    Args:
        agent_response: Response from agent.invoke() containing messages and structured_response
        node_name: Name of the node (e.g., 'planner', 'retriever')
    
    Returns:
        List of reasoning step dictionaries
    """
    reasoning_steps = []
    
    # Extract structured reasoning trace if available (final summary)
    structured = agent_response.get("structured_response")
    if structured and hasattr(structured, 'response'):
        response_obj = structured.response
        if hasattr(response_obj, 'reasoning_trace'):
            for step in response_obj.reasoning_trace:
                reasoning_steps.append({
                    "node": node_name,
                    "timestamp": datetime.now().isoformat(),
                    "action": "structured_reasoning",
                    "observation": step.observation,
                    "reasoning": step.reasoning,
                    "next_action": step.action,
                    "expected_outcome": step.expected_outcome,
                })
    
    # Extract tool calls and intermediate reasoning from message history
    messages = agent_response.get("messages", [])
    for i, msg in enumerate(messages):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Tool call
            for tc in msg.tool_calls:
                reasoning_steps.append({
                    "node": node_name,
                    "step": i,
                    "timestamp": datetime.now().isoformat(),
                    "action": "tool_call",
                    "tool_name": tc.get('name'),
                    "tool_args": tc.get('args', {}),
                })
        elif isinstance(msg, ToolMessage):
            # Tool result
            reasoning_steps.append({
                "node": node_name,
                "step": i,
                "timestamp": datetime.now().isoformat(),
                "action": "tool_result",
                "result_preview": str(msg.content)[:200],
            })
        elif isinstance(msg, AIMessage):
            # Check if this is a reasoning reflection (text response between tool calls)
            content = _get_ai_message_content(msg)
            if content and not hasattr(msg, 'tool_calls'):
                # Parse reasoning from text content
                parsed_reasoning = _parse_reasoning_text(content)
                if parsed_reasoning:
                    reasoning_steps.append({
                        "node": node_name,
                        "step": i,
                        "timestamp": datetime.now().isoformat(),
                        "action": "reasoning_reflection",
                        **parsed_reasoning
                    })
    
    return reasoning_steps


def _parse_reasoning_text(text: str) -> Dict[str, str]:
    """
    Parse reasoning reflection from text content.
    
    Looks for patterns like:
    Observation: ...
    Reasoning: ...
    Action: ...
    Expected Outcome: ...
    
    Args:
        text: Text content from AIMessage
    
    Returns:
        Dictionary with observation, reasoning, next_action, expected_outcome
        Empty dict if no reasoning pattern found
    """
    if not text or len(text.strip()) < 20:
        return {}
    
    # Look for reasoning keywords
    has_observation = "observation:" in text.lower()
    has_reasoning = "reasoning:" in text.lower()
    has_action = "action:" in text.lower()
    has_expected = "expected outcome:" in text.lower() or "expected:" in text.lower()
    
    # If we have at least 2 of the 4 components, try to parse
    if sum([has_observation, has_reasoning, has_action, has_expected]) < 2:
        return {}
    
    result = {}
    
    # Extract each component using regex
    # Observation
    obs_match = re.search(r'observation:\s*(.+?)(?=\n(?:reasoning|action|expected)|$)', text, re.IGNORECASE | re.DOTALL)
    if obs_match:
        result["observation"] = obs_match.group(1).strip()
    
    # Reasoning
    reas_match = re.search(r'reasoning:\s*(.+?)(?=\n(?:observation|action|expected)|$)', text, re.IGNORECASE | re.DOTALL)
    if reas_match:
        result["reasoning"] = reas_match.group(1).strip()
    
    # Action
    act_match = re.search(r'action:\s*(.+?)(?=\n(?:observation|reasoning|expected)|$)', text, re.IGNORECASE | re.DOTALL)
    if act_match:
        result["next_action"] = act_match.group(1).strip()
    
    # Expected Outcome
    exp_match = re.search(r'expected(?:\s+outcome)?:\s*(.+?)(?=\n(?:observation|reasoning|action)|$)', text, re.IGNORECASE | re.DOTALL)
    if exp_match:
        result["expected_outcome"] = exp_match.group(1).strip()
    
    return result if result else {}


def _persist_reasoning(session_id: str, reasoning: List[Dict[str, Any]], mode: str = "append") -> None:
    """
    Save or append reasoning trace to file for post-session analysis.
    Writes a wrapper object: { session_id, correlation_id, written_at, steps }.
    """
    if not reasoning:
        return

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    filepath = logs_dir / f"agent_reasoning_{session_id}.json"
    correlation_id = get_correlation_id()
    written_at = datetime.now().isoformat()

    try:
        if mode == "append" and filepath.exists():
            with open(filepath, "r") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                steps = existing + reasoning
            elif isinstance(existing, dict) and "steps" in existing:
                steps = existing["steps"] + reasoning
            else:
                steps = reasoning
        else:
            steps = reasoning

        data = {
            "session_id": session_id,
            "correlation_id": correlation_id,
            "written_at": written_at,
            "steps": steps,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(
            "Reasoning trace %s %s",
            "appended to" if mode == "append" else "saved to",
            filepath,
            extra={
                "extra_fields": {
                    "session_id": session_id,
                    "correlation_id": correlation_id,
                    "filepath": str(filepath),
                    "steps_count": len(steps),
                }
            },
        )
    except Exception as e:
        logger.warning(
            "Failed to persist reasoning trace: %s",
            e,
            extra={"extra_fields": {"session_id": session_id, "filepath": str(filepath)}},
        )
