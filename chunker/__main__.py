#!/usr/bin/env python3
"""Entry point for python -m chunker."""

import sys
from pathlib import Path

# Ensure chunker directory is on path so main.py can use utils, config, etc.
_chunker_dir = Path(__file__).resolve().parent
if str(_chunker_dir) not in sys.path:
    sys.path.insert(0, str(_chunker_dir))

from main import main

if __name__ == "__main__":
    main()
