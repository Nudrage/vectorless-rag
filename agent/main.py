#!/usr/bin/env python3
"""
Agent: interactive chat for querying document chunks.

Uses the 3-node pipeline architecture:
    Planner -> Retriever -> Transformer

Run: python agent/main.py   or   python main.py agent chat
"""

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from dotenv import load_dotenv
load_dotenv()

from utils.config import get_config
from utils.core import ChatAgenticFramework
from utils.errors import AgentError, ValidationError, PipelineError, LLMError
from utils.logging_config import configure_logging, get_logger


# =============================================================================
# CONSTANTS
# =============================================================================

CLI_SEPARATOR_WIDTH = 60
EXIT_COMMANDS = ("exit", "quit", "q")
HELP_COMMANDS = ("help", "?")

BANNER = f"""
{'=' * CLI_SEPARATOR_WIDTH}
Cybersecurity Document Assistant
Pipeline: Planner -> Retriever -> Transformer
{'=' * CLI_SEPARATOR_WIDTH}
""".strip()

HELP_TEXT = """
Available commands:
  help, ?     - Show this help message
  exit, quit  - Exit the application
  
Just type your question to query the documents.
"""


# =============================================================================
# SETUP
# =============================================================================

def _setup_logging() -> None:
    """Configure logging from application config."""
    config = get_config()
    configure_logging(
        level=config.logging.level,
        json_format=config.logging.json_format,
        include_correlation_id=config.logging.include_correlation_id,
        log_to_file=config.logging.log_to_file,
        log_file_path=config.logging.log_file_path,
        file_json_format=config.logging.file_json_format,
    )


def _create_framework() -> ChatAgenticFramework:
    """Initialize the chat framework with configured chunks directory."""
    config = get_config()
    return ChatAgenticFramework(chunks_dir=config.pipeline.chunks_dir)


# =============================================================================
# CHAT LOOP
# =============================================================================

def _print_banner() -> None:
    """Display the application banner."""
    print(BANNER)


def _handle_chat_response(framework: ChatAgenticFramework, session_id: str, user_input: str) -> None:
    """
    Process user input and display response.
    
    Args:
        framework: Chat framework instance
        session_id: Active session ID
        user_input: User's query text
    """
    try:
        response = framework.chat(session_id, user_input)
        print(f"\nAssistant: {response}\n")
    except ValidationError as e:
        print(f"\nValidation error: {e.message}\n")
    except PipelineError as e:
        print(f"\nPipeline error: {e.message}\n")
        logger.warning(
            "Pipeline error during chat: %s",
            e,
            extra={"extra_fields": {"session_id": session_id, "error": str(e)}},
        )
    except LLMError as e:
        print(f"\nLLM error: {e.message}\n")
        logger.warning(
            "LLM error during chat: %s",
            e,
            extra={"extra_fields": {"session_id": session_id, "error": str(e)}},
        )
    except AgentError as e:
        print(f"\nAgent error: {e.message}\n")
        logger.warning(
            "Agent error during chat: %s",
            e,
            extra={"extra_fields": {"session_id": session_id, "error": str(e)}},
        )
    except Exception as e:
        print(f"\nUnexpected error: {e}\n")
        logger.exception(
            "Unexpected error during chat",
            extra={"extra_fields": {"session_id": session_id, "error": str(e)}},
        )


def _run_chat_loop(framework: ChatAgenticFramework, session_id: str) -> None:
    """
    Main interactive chat loop.
    
    Args:
        framework: Initialized chat framework
        session_id: Active session ID
    """
    print("Ready. Type your query, 'help' for commands, or 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\nGoodbye.")
            break
        
        if not user_input:
            continue
        
        # Handle exit commands
        if user_input.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            break
        
        # Handle help commands
        if user_input.lower() in HELP_COMMANDS:
            print(HELP_TEXT)
            continue
        
        # Process chat query
        _handle_chat_response(framework, session_id, user_input)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Interactive CLI chat loop using 3-node pipeline.
    
    Initializes the framework, creates a session, and runs the chat loop.
    Ensures proper cleanup on exit.
    """
    _setup_logging()
    global logger
    logger = get_logger(__name__)
    
    _print_banner()
    print("\nInitializing framework...")
    
    framework = None
    session_id = None
    
    try:
        framework = _create_framework()
        session_id = framework.create_session()
        logger.info("Session created: %s", session_id)
        _run_chat_loop(framework, session_id)
    finally:
        # Cleanup session if it was created
        if framework and session_id:
            try:
                framework.delete_session(session_id)
                logger.info("Session ended: %s", session_id)
            except Exception as e:
                logger.warning("Failed to cleanup session %s: %s", session_id, e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        # Logger may not be initialized if setup failed
        logging_module = __import__("logging")
        logging_module.exception("Fatal error: %s", e)
        sys.exit(1)
