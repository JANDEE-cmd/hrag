import os
import json
import time
import logging
from typing import List, Dict
import numpy as np

import litellm

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

# Most hosted embedding APIs cap batch size; 100 is a safe, broadly
# compatible default across providers routed through LiteLLM.
BATCH_SIZE = 100


class BaseVectorStore:
    def __init__(self, config: dict, mode: str):
        self.mode = mode
        self.config = config[mode]
        self.vector_db_type = self.config['vector_db']
        self.embedding_model_name = self.config['embedding_model']
        self.index_path = "vector_index.bin"
        self.metadata_path = "vector_metadata.json"

        self.api_key = None
        if self.mode == "offline":
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded offline embedder: {self.embedding_model_name}")
        else:
            env_var = self.config.get('api_key_env_var')
            if env_var:
                self.api_key = os.getenv(env_var)
                if not self.api_key:
                    raise ValueError(
                        f"System Error: {env_var} environment variable is not set!"
                    )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using either the local offline embedder or LiteLLM
        (works with any embedding-capable provider LiteLLM supports —
        OpenAI, Gemini, Cohere, Azure, Voyage, etc. — based on
        `embedding_model`, e.g. 'gemini/gemini-embedding-001',
        'openai/text-embedding-3-small').
        """
        if self.mode == "offline":
            logger.info(f"Embedding {len(texts)} texts offline...")
            return np.asarray(
                self.embedder.encode(texts, show_progress_bar=True), dtype=np.float32
            )
        else:
            return self._embed_texts_online_batched(texts)

    def _embed_texts_online_batched(self, texts: List[str]) -> np.ndarray:
        """
        Batched embedding via LiteLLM. Significantly reduces API calls and
        cost compared to one-per-chunk, and works the same way regardless
        of which provider `embedding_model` points at.
        """
        embeddings = []

        logger.info(f"Embedding {len(texts)} chunks with batches of {BATCH_SIZE}...")

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

            try:
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")

                result = litellm.embedding(
                    model=self.embedding_model_name,
                    input=batch,
                    api_key=self.api_key,
                )

                batch_embeddings = [item["embedding"] for item in result.data]
                embeddings.extend(batch_embeddings)

                logger.info(f"✓ Batch {batch_num}/{total_batches} complete")

                if i + BATCH_SIZE < len(texts):
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to embed batch {batch_num}: {e}")
                raise RuntimeError(f"Embedding batch {batch_num} failed: {str(e)}")

        logger.info(f"Successfully embedded all {len(texts)} texts")
        return np.array(embeddings, dtype=np.float32)

    def build_index(self, chunked_data: List[Dict]):
        """Builds and saves the vector database with error handling."""
        if not chunked_data:
            raise ValueError("No chunked data provided to build index")

        texts = [item['content'] for item in chunked_data]

        logger.info(f"Encoding {len(texts)} texts to vectors...")

        try:
            embeddings = self.embed_texts(texts)
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise

        if embeddings.shape[0] != len(chunked_data):
            raise RuntimeError(
                f"Embedding count mismatch: got {embeddings.shape[0]}, "
                f"expected {len(chunked_data)}"
            )

        logger.info(f"Building {self.vector_db_type} index...")

        if self.vector_db_type == "faiss":
            self._build_faiss(embeddings, chunked_data)
        else:
            logger.error(f"Vector DB '{self.vector_db_type}' is not supported yet")
            raise ValueError(f"Unsupported vector_db type: {self.vector_db_type}")

    def _build_faiss(self, embeddings: np.ndarray, chunked_data: List[Dict]):
        """
        Implementation for FAISS index creation and persistence.
        Includes validation and error handling.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        if embeddings.shape[0] != len(chunked_data):
            raise ValueError("Embedding and data mismatch")

        dimension = embeddings.shape[1]
        logger.info(f"Creating FAISS index (dimension={dimension}, vectors={embeddings.shape[0]})...")

        try:
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            logger.info(f"Index created with {index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise

        # Save with atomic write (avoid partial files on crash)
        try:
            temp_index_path = self.index_path + ".tmp"
            temp_metadata_path = self.metadata_path + ".tmp"

            faiss.write_index(index, temp_index_path)
            logger.debug(f"FAISS index written to {temp_index_path}")

            with open(temp_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(chunked_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Metadata written to {temp_metadata_path}")

            if os.path.exists(self.index_path):
                os.rename(self.index_path, self.index_path + ".bak")
            if os.path.exists(self.metadata_path):
                os.rename(self.metadata_path, self.metadata_path + ".bak")

            os.rename(temp_index_path, self.index_path)
            os.rename(temp_metadata_path, self.metadata_path)

            logger.info(f"✓ FAISS index and metadata saved ({index.ntotal} vectors)")

            if os.path.exists(self.index_path + ".bak"):
                os.remove(self.index_path + ".bak")
            if os.path.exists(self.metadata_path + ".bak"):
                os.remove(self.metadata_path + ".bak")

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            for f in [temp_index_path, temp_metadata_path]:
                if os.path.exists(f):
                    os.remove(f)
            raise