This is a comprehensive, production-ready `README.md` file. It goes deep into the architecture, configuration parameters, and detailed usage examples.

You can copy the entire block below and save it exactly as `README.md` in the root directory of your project.

---

```markdown
# 🚀 Hybrid RAG CLI (hrag)

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-green.svg)
![Architecture](https://img.shields.io/badge/architecture-Hybrid%20Search%20(FAISS%20%2B%20BM25)-orange.svg)

**hrag** is a robust, terminal-based Hybrid Retrieval-Augmented Generation (RAG) Command Line Interface. It is designed to provide highly accurate, LLM-generated answers based on your private documents by leveraging a state-of-the-art Hybrid Search architecture.

Instead of relying solely on Vector Embeddings (which can struggle with exact keyword matching) or purely on Lexical Search (which misses semantic context), **hrag** combines both using **Reciprocal Rank Fusion (RRF)**. The result is a seamless, ChatGPT-like streaming experience directly in your terminal, grounded entirely in your own data.

## ✨ Core Architecture & Features

- **Hybrid Search Engine:** 
  - **Dense Retrieval:** FAISS (Facebook AI Similarity Search) for semantic similarity.
  - **Sparse Retrieval:** BM25 (Best Matching 25) for exact keyword and structural matching.
  - **Re-ranking:** Reciprocal Rank Fusion (RRF) algorithm to elegantly combine and re-rank results from both engines.
- **Dual Execution Modes:**
  - **`Online` Mode:** Connects to cloud providers via LiteLLM (Google Gemini, OpenAI, Anthropic).
  - **`Offline` Mode:** Runs 100% locally with zero internet connection (e.g., Ollama, Local HuggingFace Embeddings).
- **Real-Time Streaming UI:** Provides a responsive, typewriter-effect output in the console.
- **Robust Schema Validation:** YAML configuration is strictly validated before execution to prevent runtime crashes.
- **End-to-End Diagnostics:** Built-in commands to verify system health, computational backends (CPU/GPU), and environment variables.

---

## 🛠️ Prerequisites

Before installing the CLI, ensure your system meets the following requirements:
- **Python:** Version 3.11 or higher.
- **Git:** To clone the repository.
- **C++ Build Tools (Windows Only):** Required for compiling FAISS and BM25 dependencies.
- **API Key:** A valid API key for your chosen provider (e.g., Google Gemini) if operating in `online` mode.

---

## 📥 Installation

It is strongly recommended to install the tool within an isolated Python Virtual Environment (`venv` or `conda`).

### 1. Clone the Repository
```bash
git clone [https://github.com/JANDEE-cmd/hrag.git](https://github.com/JANDEE-cmd/hrag.git)
cd hrag

```

### 2. Set Up Virtual Environment

**Windows:**

```cmd
python -m venv venv
.\venv\Scripts\activate

```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install the CLI Package

Install the project in editable mode so it registers the `hrag` command globally within your environment.

```bash
pip install .

```

### 4. Verify Installation

```bash
hrag --version

```

*Expected Output: `rag-cli version 0.1.0*`

---

## 🚀 Complete User Guide

Follow this step-by-step guide to run your first Hybrid RAG pipeline.

### Step 1: Initialize the Project Workspace

Navigate to the directory where you want to store your data and configuration, then initialize the CLI.

```bash
mkdir my-rag-workspace
cd my-rag-workspace

# Generate the default configuration file
hrag init --template online

```

This will create a `config.yaml` file in your current directory.

### Step 2: Configure Environment Variables

The system requires authentication for cloud models. You must set the API key defined in your config (default: `GEMINI_API_KEY`).

**Option A: Using the Terminal (Temporary)**

* **Windows (PowerShell):** `$env:GEMINI_API_KEY="your_actual_api_key_here"`
* **Windows (CMD):** `set GEMINI_API_KEY=your_actual_api_key_here`
* **Mac/Linux:** `export GEMINI_API_KEY="your_actual_api_key_here"`

**Option B: Using a `.env` File (Persistent - Recommended)**
Create a file named `.env` in the same directory as your `config.yaml` and add:

```text
GEMINI_API_KEY=your_actual_api_key_here

```

*(Note for Windows users: Avoid using the `echo` command to create the `.env` file as it may introduce UTF-16 encoding or unwanted quotes. Please use a text editor like Notepad or VSCode).*

### Step 3: Run System Diagnostics

Verify that the CLI can read your configuration and API keys.

```bash
hrag diagnostics

```

*Expected Output:*

```text
--- System Diagnostics Report ---
Python Runtime   : 3.11.x
Compute Backend  : CPU
Environment Setup: PASS
---------------------------------

```

### Step 4: Prepare Your Data

Create a directory named `data` (as specified in the config) and place your text documents inside.

```bash
mkdir data

```

Create a test file `data/quantum.txt` with some domain-specific knowledge:

```text
Quantum computing utilizes qubits, which can exist in a state of 0, 1, or both simultaneously due to superposition. Unlike classical bits, qubits leverage quantum decoherence, a phenomenon where environmental noise causes calculation errors.

```

### Step 5: Execute Data Ingestion

This command reads all documents in the `data/` folder, chunks them, generates embeddings via the LLM provider, and builds the dual FAISS + BM25 index.

```bash
hrag ingest

```

*Expected Output:*

```text
--- Starting Data Ingestion Pipeline (Mode: ONLINE, Embedding: gemini/gemini-embedding-001) ---
Scanning directory: ./data
Total chunks created: 1
Building Vector Database (force=False, workers=1)...
--- Data Ingestion Pipeline: PASS ---

```

### Step 6: Query the System

Ask a question based on the ingested documents. The system will stream the generated response to your terminal.

```bash
hrag ask "What is the difference between classical bits and qubits?"

```

---

## ⚙️ Configuration Reference (`config.yaml`)

The `config.yaml` file is the brain of the `hrag` CLI. It undergoes strict Pydantic validation before any command is executed.

```yaml
project_name: "my-first-rag"  # Name of your project
mode: "online"                # Operational mode: 'online' or 'offline'
system_prompt: "You are a helpful research assistant. Use the provided context to answer the question accurately."

# Data Processing Parameters
data:
  chunk_size: 1000            # Maximum character length per chunk
  chunk_overlap: 200          # Character overlap between chunks to preserve context
  docs_path: "./data"         # Relative or absolute path to your document directory

# Online Mode Parameters (Cloud APIs)
online:
  llm_model: "gemini/gemini-2.5-flash"         # LiteLLM compatible model string for generation
  embedding_model: "gemini/gemini-embedding-001" # LiteLLM compatible model string for embeddings
  vector_db: "faiss"                           # Vector storage backend
  api_key_env_var: "GEMINI_API_KEY"            # The environment variable containing your API key
  urls:                                        # (Feature in development) Web pages to scrape and ingest
    - "[https://en.wikipedia.org/wiki/Artificial_intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence)"

# Offline Mode Parameters (Local Models)
offline:
  llm_model: "llama3.2:1b"           # Local model name (requires Ollama to be running)
  embedding_model: "all-MiniLM-L6-v2" # Local sentence-transformers model
  vector_db: "faiss"

```

---

## ⚠️ Common Troubleshooting

### 1. `Configuration Error: config.yaml failed validation.`

**Cause:** The `config.yaml` is missing required fields or has incorrect data types.
**Solution:** Ensure `data.docs_path` exists in your config. Run `hrag validate` to see the exact schema violation.

### 2. `Warning: Missing required environment variables: ['GEMINI_API_KEY']`

**Cause:** The CLI cannot find the API key specified in the config.
**Solution:** If using a `.env` file on Windows, ensure it is saved with UTF-8 encoding and does not contain leading/trailing spaces or quotation marks around the key.

### 3. `NotFoundError: GeminiException - code: 404...` during Ingestion

**Cause:** The embedding model specified in `config.yaml` (e.g., `text-embedding-004`) is not accessible or deprecated for your API version.
**Solution:** Open `config.yaml` and change `embedding_model` to a stable fallback like `"gemini/gemini-embedding-001"`.

---

## 📝 License

This project is open-source and available under the MIT License.

```

```
