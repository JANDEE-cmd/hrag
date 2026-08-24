import os
import json
import time
import logging
from typing import List, Dict
import numpy as np
import litellm
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

# Most hosted embedding APIs cap batch size; 100 is a safe, broadly
# compatible default across providers routed through LiteLLM.
BATCH_SIZE = 100


class HybridSearchEngine:
    """
    Combines Semantic Search (Vector Embeddings) and Keyword Search (BM25)
    using Reciprocal Rank Fusion (RRF) for robust and accurate retrieval.
    """
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        # Tokenize corpus for BM25 keyword search
        tokenized_corpus = [chunk["content"].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform keyword-based search using BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": int(idx),
                "content": self.chunks[idx]["content"],
                "score": float(scores[idx])
            })
        return results

    def reciprocal_rank_fusion(self, vector_results: List[Dict], keyword_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        Merge vector search and keyword search results fairly 
        using the Reciprocal Rank Fusion (RRF) algorithm.
        """
        rrf_scores = {}
        
        # Aggregate ranks from Vector Search
        for rank, res in enumerate(vector_results):
            chunk_id = res["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"chunk_id": chunk_id, "content": res["content"], "score": 0.0}
            rrf_scores[chunk_id]["score"] += 1.0 / (k + (rank + 1))
            
        # Aggregate ranks from Keyword Search (BM25)
        for rank, res in enumerate(keyword_results):
            chunk_id = res["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"chunk_id": chunk_id, "content": res["content"], "score": 0.0}
            rrf_scores[chunk_id]["score"] += 1.0 / (k + (rank + 1))
            
        # Sort combined results by RRF score descending
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results


class BaseVectorStore:
    """
    Manages vector embeddings, index building, and hybrid search retrieval 
    across local (offline) and cloud (online) environments.
    """
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
        """Embed texts using either the local offline embedder or LiteLLM."""
        if self.mode == "offline":
            logger.info(f"Embedding {len(texts)} texts offline...")
            return np.asarray(
                self.embedder.encode(texts, show_progress_bar=True), dtype=np.float32
            )
        else:
            return self._embed_texts_online_batched(texts)

    def _embed_texts_online_batched(self, texts: List[str]) -> np.ndarray:
        """Batched embedding via LiteLLM to reduce API calls and costs."""
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
        """Implementation for FAISS index creation and atomic persistence."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        dimension = embeddings.shape[1]
        logger.info(f"Creating FAISS index (dimension={dimension}, vectors={embeddings.shape[0]})...")

        try:
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            logger.info(f"Index created with {index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise

        temp_index_path = self.index_path + ".tmp"
        temp_metadata_path = self.metadata_path + ".tmp"

        try:
            faiss.write_index(index, temp_index_path)
            with open(temp_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(chunked_data, f, ensure_ascii=False, indent=2)

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
            for path in [temp_index_path, temp_metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform Hybrid Search (Vector Semantic Search + BM25 Keyword Search)
        combined via Reciprocal Rank Fusion (RRF).
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed.")

        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Vector index or metadata not found. Please run 'hrag ingest' first.")

        # Load metadata (chunks)
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # 1. Vector Semantic Search
        query_vector = self.embed_texts([query])
        index = faiss.read_index(self.index_path)
        
        # Fetch more candidates for better RRF blending
        fetch_k = min(top_k * 3, len(chunks))
        distances, indices = index.search(query_vector, fetch_k)

        vector_results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            vector_results.append({
                "chunk_id": int(idx),
                "content": chunks[idx]["content"],
                "score": float(score)
            })

        # 2. Keyword Search (BM25)
        hybrid_engine = HybridSearchEngine(chunks)
        keyword_results = hybrid_engine.keyword_search(query, top_k=fetch_k)

        # 3. Combine using RRF
        final_results = hybrid_engine.reciprocal_rank_fusion(vector_results, keyword_results)

        # Return top_k results
        return final_results[:top_k]