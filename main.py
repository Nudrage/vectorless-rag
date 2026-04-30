#!/usr/bin/env python3
"""
Parent entry point: dispatches to Chunker or Agent sub-modules.

Usage:
  python main.py chunk <pdf_path>       -- Process PDF (Chunker)
  python main.py agent chat             -- Interactive CLI chat (Agent)
  python main.py agent serve [--port N] -- Start FastAPI chat server (Agent)

Run from project root.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CHUNKER_DIR = PROJECT_ROOT / "chunker"
AGENT_DIR = PROJECT_ROOT / "agent"


def run_chunk(args: argparse.Namespace) -> int:
    """Run Chunker: process PDF."""
    chunker_main_path = CHUNKER_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("chunker_main", chunker_main_path)
    chunker_main_module = importlib.util.module_from_spec(spec)
    if str(CHUNKER_DIR) not in sys.path:
        sys.path.insert(0, str(CHUNKER_DIR))
    spec.loader.exec_module(chunker_main_module)
    original_argv = sys.argv
    sys.argv = [original_argv[0], args.pdf]
    try:
        chunker_main_module.main()
        return 0
    except SystemExit as e:
        return e.code if e.code is not None else 0
    finally:
        sys.argv = original_argv


def run_agent_chat(args: argparse.Namespace) -> int:
    """Run Agent: interactive CLI chat."""
    if str(AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DIR))
    import logging
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    try:
        from utils.core import ChatAgenticFramework
        framework = ChatAgenticFramework(
            api_key=getattr(args, 'api_key', None),
            model=getattr(args, 'model', None),
            chunks_dir=getattr(args, 'chunks_dir', 'output'),
        )
        session_id = framework.create_session()
        print("=" * 60)
        print("Cybersecurity Document Assistant")
        print("=" * 60)
        print("\nReady. Type your query, or 'exit' to quit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\nGoodbye.")
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break
            response = framework.chat(session_id, user_input)
            print("\nAssistant:", response, "\n")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        logging.getLogger(__name__).error(f"Error: {e}", exc_info=getattr(args, 'verbose', False))
        return 1


def run_agent_serve(args: argparse.Namespace) -> int:
    """Run Agent: FastAPI server."""
    if str(AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DIR))
    from server import serve
    serve(host=getattr(args, 'host', '0.0.0.0'), port=getattr(args, 'port', 8000))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Context Service Management - Chunker and Agent'
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommand')

    chunk_parser = subparsers.add_parser('chunk', help='Process PDF (Chunker)')
    chunk_parser.add_argument('pdf', help='Path to the PDF file to process')
    chunk_parser.set_defaults(func=run_chunk)

    agent_parser = subparsers.add_parser('agent', help='Query chunks (Agent chat or serve)')
    agent_sub = agent_parser.add_subparsers(dest='agent_command', required=True, help='Agent mode')

    chat_parser = agent_sub.add_parser('chat', help='Interactive CLI chat')
    chat_parser.add_argument('--chunks-dir', default='output', help='Chunks directory')
    chat_parser.add_argument('--api-key', help='LLM API key (or OPENAI_API_KEY/GOOGLE_API_KEY env)')
    chat_parser.add_argument('--model', help='LLM model name (defaults based on provider)')
    chat_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    chat_parser.set_defaults(func=run_agent_chat)

    serve_parser = agent_sub.add_parser('serve', help='Start FastAPI chat server')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port (default: 8000)')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0)')
    serve_parser.add_argument('--chunks-dir', default='output', help='Chunks directory')
    serve_parser.set_defaults(func=run_agent_serve)

    args = parser.parse_args()
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
