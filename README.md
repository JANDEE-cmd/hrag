# Hybrid RAG

Hybrid RAG is a Python CLI for academic research workflows that supports both offline and online retrieval/answer pipelines.

## Features

- Generate a starter `config.yaml`
- Validate configuration and environment
- Ingest local `.txt` and `.md` documents
- Build a FAISS vector index
- Retrieve relevant chunks and answer questions
- Run in offline mode with local Ollama models or online mode with Gemini

## Requirements

- Python 3.9+
- A virtual environment is recommended
- For offline mode: Ollama installed and running locally
- For online mode: a valid `GEMINI_API_KEY` environment variable

## Install in a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If you only want the Python dependencies without installing the local package itself, you can run:

```bash
pip install -r requirements.txt
```

## Environment variables

Set the Gemini API key before running online mode:

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

macOS/Linux:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

## Quick start

1. Create a default config:

```bash
hrag init --template offline
```

2. Ingest documents from the `data` directory:

```bash
hrag ingest --config config.yaml
```

3. Ask a question:

```bash
hrag ask --config config.yaml "What is the main idea of this research?"
```

## Offline mode notes

For offline mode, make sure Ollama is running:

```bash
ollama serve
ollama pull llama3.2:1b
```

## Project structure

- `hybrid_rag/` — Python package source
- `config.yaml` — runtime configuration
- `data/` — source documents
- `vector_index.bin` — generated FAISS index
- `vector_metadata.json` — metadata for indexed chunks

## Troubleshooting

- If `hrag` is not recognized after installation, run the environment activation again and verify that `pip install -e .` succeeded.
- If the vector database is missing, run `hrag ingest` before `hrag ask`.
- If Ollama is not available, switch to online mode or install Ollama locally.
