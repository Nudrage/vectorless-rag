#!/usr/bin/env python3
"""
FastAPI server for the 3-node pipeline document chat framework.

Pipeline: Planner -> Retriever -> Transformer

Run: python agent/server.py   or   python main.py agent serve
Endpoints:
    POST   /sessions                      - Create new session
    POST   /sessions/{session_id}/chat    - Send message
    GET    /sessions/{session_id}/history - Get conversation history
    DELETE /sessions/{session_id}         - Delete session
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from utils.config import get_config
from utils.core import ChatAgenticFramework
from utils.errors import (
    AgentError,
    DocumentNotFoundError,
    PipelineError,
    ValidationError,
)
from utils.logging_config import (
    configure_logging,
    get_correlation_id,
    get_logger,
    new_correlation_id,
    set_correlation_id,
)
from utils.models import (
    ChatRequest,
    ChatResponse,
    CreateSessionResponse,
    HistoryResponse,
    MessageItem,
)

# Configure logging from config (before get_logger use)
_config = get_config()
configure_logging(
    level=_config.logging.level,
    json_format=_config.logging.json_format,
    include_correlation_id=_config.logging.include_correlation_id,
    log_to_file=_config.logging.log_to_file,
    log_file_path=_config.logging.log_file_path,
    file_json_format=_config.logging.file_json_format,
)
logger = get_logger(__name__)

CHUNKS_DIR = _config.pipeline.chunks_dir
app = FastAPI(
    title="Cybersecurity Document Chat API",
    version="2.0.0",
    description="3-node pipeline: Planner -> Retriever -> Transformer",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Set correlation ID from X-Request-ID header or generate one."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or new_correlation_id()
        set_correlation_id(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)

framework: Optional[ChatAgenticFramework] = None


def get_framework() -> ChatAgenticFramework:
    global framework
    if framework is None:
        framework = ChatAgenticFramework(chunks_dir=CHUNKS_DIR)
    return framework


# -----------------------------------------------------------------------------
# Exception handlers
# -----------------------------------------------------------------------------


def _request_context(request: Request) -> dict:
    """Build extra_fields for exception logs: correlation_id and session_id when present."""
    ctx = {"correlation_id": get_correlation_id()}
    session_id = request.path_params.get("session_id") if request.path_params else None
    if session_id:
        ctx["session_id"] = session_id
    return ctx


@app.exception_handler(ValidationError)
def handle_validation_error(request: Request, exc: ValidationError):
    logger.warning(
        "Validation error",
        extra={"extra_fields": {"message": exc.message, **_request_context(request)}},
    )
    return fastapi_json_response(422, "Validation error", str(exc))


@app.exception_handler(DocumentNotFoundError)
def handle_document_not_found(request: Request, exc: DocumentNotFoundError):
    logger.warning(
        "Document not found",
        extra={"extra_fields": {"message": exc.message, **_request_context(request)}},
    )
    return fastapi_json_response(404, "Document not found", str(exc), getattr(exc, "details", None))


@app.exception_handler(PipelineError)
def handle_pipeline_error(request: Request, exc: PipelineError):
    logger.exception(
        "Pipeline error",
        extra={"extra_fields": _request_context(request)},
    )
    return fastapi_json_response(503, "Pipeline error", str(exc))


@app.exception_handler(AgentError)
def handle_agent_error(request: Request, exc: AgentError):
    logger.exception(
        "Agent error",
        extra={"extra_fields": _request_context(request)},
    )
    return fastapi_json_response(500, "Agent error", str(exc), getattr(exc, "details", None))


def fastapi_json_response(status_code: int, error: str, detail: str, details: Optional[Dict[str, Any]] = None):
    from fastapi.responses import JSONResponse
    body = {"error": error, "detail": detail}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    """Create a new chat session. Returns session_id."""
    fw = get_framework()
    session_id = fw.create_session()
    return CreateSessionResponse(session_id=session_id)


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, body: ChatRequest) -> ChatResponse:
    """Send a message in a session and get the agent response."""
    fw = get_framework()
    if not fw.has_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        response_text = fw.chat(session_id, body.message)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    metadata = fw.get_session_metadata(session_id)
    return ChatResponse(response=response_text, metadata=metadata)


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str) -> HistoryResponse:
    """Get conversation history for a session."""
    fw = get_framework()
    if not fw.has_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    history = fw.get_history(session_id)
    from langchain_core.messages import HumanMessage, AIMessage
    items = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            items.append(MessageItem(type="human", content=content))
        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                content = " ".join(str(p) for p in content)
            elif not isinstance(content, str):
                content = str(content)
            items.append(MessageItem(type="ai", content=content))
    return HistoryResponse(messages=items)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, str]:
    """End a session (remove from server memory)."""
    fw = get_framework()
    if not fw.has_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    fw.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    serve(port=port)
