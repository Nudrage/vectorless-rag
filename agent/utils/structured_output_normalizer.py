"""
Normalize LLM tool-call args so providers that send nested JSON as strings (e.g. Gemini)
still validate against our Pydantic schemas.

- Top-level: args["response"] may be a JSON string → parse to object.
- Nested: response.retrieval_plan may be a string or missing; reasoning_trace may be
  list of strings. Coerce to the shapes expected by PlannerResponse / RetrieverResponse.
"""

from __future__ import annotations

import json
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)

# Tool names that have a top-level "response" field which some providers send as a JSON string.
STRUCTURED_RESPONSE_TOOLS = ("PlannerResponse", "RetrieverResponse")


def _normalize_retrieval_plan(value: Any) -> dict[str, Any]:
    """Coerce retrieval_plan to { use_case, reasoning } when LLM sends string or omits."""
    if isinstance(value, dict) and "use_case" in value:
        return value
    if isinstance(value, str):
        s = value.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict) and "use_case" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {"use_case": "generic_qa_task", "reasoning": s}
    return {"use_case": "generic_qa_task", "reasoning": ""}


def _normalize_reasoning_trace(items: Any) -> list[dict[str, Any]]:
    """Coerce reasoning_trace to list of { observation, reasoning, action, expected_outcome }."""
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and "observation" in item:
            out.append(item)
            continue
        if isinstance(item, str):
            out.append({
                "observation": item,
                "reasoning": "",
                "action": "",
                "expected_outcome": "",
            })
            continue
        out.append({
            "observation": str(item),
            "reasoning": "",
            "action": "",
            "expected_outcome": "",
        })
    return out


def _normalize_planner_proceed(obj: dict[str, Any]) -> dict[str, Any]:
    """Ensure retrieval_plan is an object and reasoning_trace is list of objects."""
    out = dict(obj)
    if out.get("status") != "proceed":
        return out
    if "retrieval_plan" not in out or not isinstance(out.get("retrieval_plan"), dict):
        out["retrieval_plan"] = _normalize_retrieval_plan(out.get("retrieval_plan"))
        logger.debug(
            "Structured output normalizer coerced field",
            extra={
                "extra_fields": {
                    "tool_name": "PlannerResponse",
                    "field": "retrieval_plan",
                    "action": "coerced",
                }
            },
        )
    if "reasoning_trace" in out and isinstance(out["reasoning_trace"], list):
        original = out["reasoning_trace"]
        out["reasoning_trace"] = _normalize_reasoning_trace(out["reasoning_trace"])
        if original != out["reasoning_trace"]:
            logger.debug(
                "Structured output normalizer coerced field",
                extra={
                    "extra_fields": {
                        "tool_name": "PlannerResponse",
                        "field": "reasoning_trace",
                        "action": "coerced",
                    }
                },
            )
    # If use_case at top level but retrieval_plan missing/invalid, build from it
    if "retrieval_plan" in out and isinstance(out["retrieval_plan"], dict):
        rp = out["retrieval_plan"]
        if "use_case" not in rp and "use_case" in out:
            rp = {**rp, "use_case": out.get("use_case", "generic_qa_task")}
        if not rp.get("reasoning") and out.get("reasoning"):
            rp = {**rp, "reasoning": out.get("reasoning", "")}
        out["retrieval_plan"] = rp
    return out


def normalize_tool_args(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize tool args: parse top-level 'response' string to object, then coerce
    nested fields (retrieval_plan, reasoning_trace) so they validate.
    """
    if tool_name not in STRUCTURED_RESPONSE_TOOLS:
        return tool_args
    raw = tool_args.get("response")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return tool_args
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        return tool_args
    # Normalize nested shape for planner proceed payload
    if tool_name == "PlannerResponse" and parsed.get("status") == "proceed":
        parsed = _normalize_planner_proceed(parsed)
    # RetrieverResponse proceed can have similar nesting; normalize if we add RetrieverProceedResponse fields later
    return {**tool_args, "response": parsed}


def apply_llm_args_normalizer() -> None:
    """
    Patch OutputToolBinding.parse so that tool_args are normalized before validation.
    Call once at startup (e.g. when planner or retriever node is first used).
    """
    from langchain.agents.structured_output import OutputToolBinding

    _original_parse = OutputToolBinding.parse

    def patched_parse(self: OutputToolBinding, tool_args: dict[str, Any]) -> Any:
        name = getattr(self.tool, "name", None) or ""
        normalized = normalize_tool_args(name, tool_args)
        return _original_parse(self, normalized)

    OutputToolBinding.parse = patched_parse


# Apply patch on import so any create_agent(..., ToolStrategy(...)) benefits.
apply_llm_args_normalizer()
