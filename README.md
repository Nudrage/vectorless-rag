# Context Service Management

Document processing and RAG (retrieval-augmented generation) pipeline: chunk PDFs with hierarchy preservation, then query them via an agent.

## Layout

- **`chunker/`** – PDF chunker: extract text, detect headers by font, merge body/table-aware, save to `output/chunks/<document_name>/`
- **`agent/`** – RAG agent (LangGraph + Gemini): loads chunks from `output/chunks/`, answers queries
- **`data/`** – Input PDFs and other source files
- **`output/chunks/`** – Chunk output (created when you run the chunker)
- **`main.py`** – Root CLI: dispatches to chunker or agent

## Requirements

- Python 3.10+
- Install: `pip install -r requirements.txt`  
  (Or per app: `pip install -r chunker/requirements.txt` or `pip install -r agent/requirements.txt`)

## Usage

From the project root:

```bash
# Process a PDF (chunker) – writes to output/chunks/<document_name>/
python main.py chunk path/to/document.pdf

# Query chunks (agent) – uses output/chunks by default
python main.py agent "Your question here"
python main.py agent --chunks-dir output/chunks "Your question"
```

Optional: set `GEMINI_API_KEY` in `.env` (or pass `--api-key`) for the agent.
