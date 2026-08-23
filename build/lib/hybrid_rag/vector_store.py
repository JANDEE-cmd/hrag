from typing import List, Dict
import numpy as np

class BaseVectorStore:
    def __init__(self, config: dict, mode: str):
        self.mode = mode
        self.config = config[mode]
        self.vector_db_type = self.config['vector_db']
        self.embedding_model_name = self.config['embedding_model']
        
        if self.mode == "offline":
            print(f"Loading embedding model: {self.embedding_model_name}...")
            # โหลด Model แบบ Offline
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Converts text into vector embeddings."""
        if self.mode == "offline":
            return self.embedder.encode(texts, show_progress_bar=True)
        else:
            raise NotImplementedError("Online API embedding is not yet implemented.")

    def build_index(self, chunked_data: List[Dict]):
        """Builds the vector database."""
        texts = [item['content'] for item in chunked_data]
        print("Encoding texts to vectors...")
        embeddings = self.embed_texts(texts)
        
        if self.vector_db_type == "faiss":
            self._build_faiss(embeddings, chunked_data)
        else:
            print(f"Warning: Vector DB '{self.vector_db_type}' save logic is not implemented yet.")

    def _build_faiss(self, embeddings: np.ndarray, chunked_data: List[Dict]):
        """Simple FAISS implementation."""
        import faiss
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        print(f"FAISS index built successfully with {index.ntotal} vectors.")