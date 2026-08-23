import os
import json
import time
import logging
import numpy as np
from dotenv import load_dotenv

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

load_dotenv()
logger = logging.getLogger(__name__)

# litellm prints its own (fairly noisy) request/response logs by default.
litellm.suppress_debug_info = True

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Use the provided context to answer "
    "the question accurately. If the context does not contain the answer, say so "
    "instead of guessing."
)

# Errors worth retrying: transient network/availability issues, not things
# like bad prompts or auth failures.
RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, ServiceUnavailableError, Timeout)

# Default local Ollama server. Can be overridden with the OLLAMA_API_BASE
# env var (litellm reads this automatically) if Ollama runs elsewhere.
DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"


def retry_with_backoff(max_attempts: int = 3, base_delay: float = 1.0):
    """
    Decorator for exponential backoff retry logic.

    Retries on transient LiteLLM/provider errors (rate limits, timeouts,
    connection issues, temporary unavailability). Non-retryable errors
    (bad auth, invalid model, etc.) are raised immediately.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_ERRORS as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                except Exception:
                    # Don't retry for non-transient errors (auth, bad model, etc.)
                    raise
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


class ChatEngine:
    """
    RAG chat engine backed by LiteLLM.

    LiteLLM gives us a single interface across providers: point `llm_model`
    at "ollama/llama3.2:1b" for local models, or "gemini/gemini-3.5-flash",
    "openai/gpt-4o", "anthropic/claude-3-5-sonnet-latest", "azure/<deployment>",
    etc. for hosted APIs -- same `generate()` call either way.
    """

    def __init__(self, config: dict, mode: str):
        self.mode = mode
        # Keep the system prompt (top-level key) before narrowing to the
        # mode-specific section, otherwise --system-prompt silently no-ops.
        self.system_prompt = config.get('system_prompt') or DEFAULT_SYSTEM_PROMPT

        self.config = config[mode]
        self.llm_model = self._normalize_model_name(self.config['llm_model'])
        self.embedding_model = self.config.get('embedding_model')
        self.vector_db_type = self.config['vector_db']
        self.index_path = "vector_index.bin"
        self.metadata_path = "vector_metadata.json"

        # Resolve the API key from whichever env var the config points at.
        # Passed explicitly to every litellm call so behavior doesn't depend
        # on the provider's "standard" env var name (e.g. OPENAI_API_KEY).
        self.api_key = None
        env_var = self.config.get('api_key_env_var')
        if env_var:
            self.api_key = os.getenv(env_var)
            if not self.api_key:
                raise ValueError(
                    f"System Error: {env_var} environment variable is not set!"
                )

        # Local Ollama doesn't need a key but does need a reachable server.
        self.api_base = DEFAULT_OLLAMA_API_BASE if self.mode == "offline" else None

        logger.info(f"Initializing ChatEngine in {mode} mode with model: {self.llm_model}")

        self._load_local_db()

        if self.mode == "offline":
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model)
            logger.info(f"Loaded embedder: {self.embedding_model}")

    def _normalize_model_name(self, model_name: str) -> str:
        """
        Ensure the model string carries a LiteLLM provider prefix
        (e.g. 'ollama/llama3.2:1b'). Bare offline model names from older
        configs are auto-prefixed with 'ollama/' for backwards compatibility.
        """
        if self.mode == "offline" and "/" not in model_name:
            logger.warning(
                f"llm_model '{model_name}' has no provider prefix; "
                f"assuming 'ollama/{model_name}'."
            )
            return f"ollama/{model_name}"
        return model_name

    def _load_local_db(self):
        """Loads the FAISS index and metadata from disk."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Vector database not found. Please run 'hrag ingest' first.\n"
                f"  Expected: {self.index_path}, {self.metadata_path}"
            )

        try:
            import faiss
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise

        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.metadata)} chunks")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise

    def _embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query, offline (local model) or via LiteLLM."""
        if self.mode == "offline":
            return np.asarray(self.embedder.encode([query]), dtype=np.float32)

        try:
            result = litellm.embedding(
                model=self.embedding_model,
                input=[query],
                api_key=self.api_key,
            )
            vector = result.data[0]["embedding"]
            return np.array([vector], dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to embed query via LiteLLM: {e}")
            raise

    def retrieve(self, query: str, top_k: int = 2) -> str:
        """
        Retrieve relevant context chunks for a query.

        Returns a meaningful placeholder message (instead of "") if no
        relevant chunks are found, so downstream prompts stay coherent.
        """
        logger.debug(f"Retrieving top {top_k} chunks for query: {query[:100]}...")

        query_vector = self._embed_query(query)
        distances, indices = self.index.search(query_vector, top_k)

        logger.debug(f"Search returned indices: {indices[0]}, distances: {distances[0]}")

        contexts = []
        for idx in indices[0]:
            if idx != -1:
                try:
                    chunk = self.metadata[idx]
                    contexts.append(chunk["content"])
                except (KeyError, IndexError) as e:
                    logger.warning(f"Skipping chunk {idx}: {e}")
                    continue

        if not contexts:
            warning = (
                "[⚠️ No relevant context found in the knowledge base. "
                "Providing answer based on model training data.]"
            )
            logger.warning("No relevant chunks found for query")
            return warning

        logger.info(f"Retrieved {len(contexts)} relevant chunks")
        return "\n\n---\n\n".join(contexts)

    def generate(self, query: str, context: str) -> str:
        """Constructs prompt (with system instructions) and calls the LLM via LiteLLM."""
        logger.debug(f"Generating response for query: {query[:100]}...")

        user_content = (
            f"Context Information:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        return self._call_llm(messages)

    @retry_with_backoff(max_attempts=3, base_delay=1.5)
    def _call_llm(self, messages: list) -> str:
        """
        Single LiteLLM completion call, used for both offline (Ollama) and
        online (any hosted provider) modes.
        """
        try:
            logger.debug(f"Calling LiteLLM with model: {self.llm_model}")

            call_kwargs = dict(
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
                api_key=self.api_key,
            )

            if self.mode == "offline":
                call_kwargs["api_base"] = self.api_base
                # Ollama-specific options (LiteLLM forwards these straight
                # through to Ollama's /api/generate `options` payload).
                # num_gpu=0 keeps the model on CPU: small/local machines
                # commonly don't have enough VRAM, and Ollama's llama-server
                # process crashes (CUDA OOM) instead of falling back to CPU
                # on its own. Raise num_gpu (or drop it) if your GPU has
                # enough headroom for the model you're running.
                call_kwargs.update(
                    num_gpu=0,
                    num_ctx=2048,
                    num_thread=4,
                    num_predict=256,
                    repeat_penalty=1.15,
                )
            else:
                call_kwargs["max_tokens"] = 1024

            response = litellm.completion(**call_kwargs)
            logger.info(f"✓ LiteLLM call successful ({self.llm_model})")
            return response.choices[0].message.content

        except RETRYABLE_ERRORS as e:
            msg = str(e).lower()
            # cudaMalloc/OOM failures are NOT transient — retrying with the
            # same request just burns time hitting the same wall. Ollama's
            # num_gpu=0 tells it to place 0 layers on the GPU, but its
            # llama-server subprocess can still probe/reserve a CUDA context
            # on startup if a GPU is visible at all, which fails outright on
            # machines with little/no free VRAM. That's an Ollama-server-level
            # setting, not something fixable from a single API request.
            oom_signature = any(s in msg for s in (
                "cudamalloc failed", "out of memory", "cuda_host",
                "unable to allocate", "llama-server process has terminated",
            ))
            if self.mode == "offline" and oom_signature:
                logger.error(f"Ollama GPU out-of-memory: {e}")
                raise RuntimeError(
                    f"Ollama ran out of GPU memory loading '{self.llm_model}' "
                    f"(num_gpu=0 alone isn't enough to stop it probing the GPU).\n"
                    f"💡 To fix:\n"
                    f"  1. Check free VRAM: nvidia-smi\n"
                    f"  2. Restart the Ollama server fully off the GPU:\n"
                    f"       OLLAMA_LLM_LIBRARY=cpu ollama serve\n"
                    f"     (or set CUDA_VISIBLE_DEVICES=\"\" before starting it)\n"
                    f"  3. Or free VRAM / use a smaller model if you want GPU acceleration."
                ) from e
            # Otherwise: a genuine transient issue (connection refused,
            # timeout, rate limit, brief unavailability) — let the retry
            # decorator handle it as before.
            raise
        except APIError as e:
            logger.error(f"LiteLLM API Error ({self.llm_model}): {e}")
            hint = ""
            msg = str(e).lower()
            if self.mode == "offline" and ("connection" in msg or "connect" in msg):
                hint = (
                    f"\n💡 Hint: Is Ollama running? Start it with: ollama serve\n"
                    f"   (endpoint: {self.api_base})"
                )
            elif self.mode == "offline" and "not found" in msg:
                bare_name = self.llm_model.split("/", 1)[-1]
                hint = f"\n💡 Hint: Run 'ollama pull {bare_name}' to download the model."
            raise RuntimeError(f"LLM call failed ({self.llm_model}): {e}{hint}") from e
        except Exception as e:
            logger.error(f"Unexpected error calling LiteLLM: {e}")
            raise RuntimeError(f"Unexpected error calling {self.llm_model}: {e}") from e