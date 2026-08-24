# 🚀 Hybrid RAG CLI (hrag)

**hrag** is a powerful, terminal-based Hybrid Retrieval-Augmented Generation (RAG) tool. It seamlessly combines the precision of Vector Search with the keyword matching power of BM25 (via Reciprocal Rank Fusion) to deliver highly accurate, LLM-generated answers directly in your CLI.

## ✨ Key Features
- **Hybrid Search Engine:** Combines Vector Embeddings (FAISS) and Keyword Search (BM25) for superior document retrieval.
- **Real-Time Streaming:** ChatGPT-like streaming responses directly in your terminal.
- **Dual Modes (Online/Offline):** 
  - `Online`: Use cutting-edge APIs via LiteLLM (e.g., Google Gemini, OpenAI).
  - `Offline`: Run completely locally (e.g., Ollama, Local HuggingFace Embeddings).
- **Intelligent Diagnostics:** Built-in tools to check system health and API key configurations.
- **Developer Friendly:** YAML-based configuration with strict schema validation.

---

## 🛠️ Prerequisites
- **Python:** Version 3.11 or higher.
- **Git:** For cloning the repository.
- **API Key:** (For Online Mode) A valid Google Gemini API Key or other providers supported by LiteLLM.

---

## 📥 Installation

It is highly recommended to use a Python Virtual Environment (`venv` or `conda`) to avoid dependency conflicts.

```bash
# 1. Clone the repository
git clone [https://github.com/JANDEE-cmd/hrag.git](https://github.com/JANDEE-cmd/hrag.git)
cd hrag

# 2. Create and activate a virtual environment (Optional but recommended)
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 3. Install the CLI tool globally within the environment
pip install .
