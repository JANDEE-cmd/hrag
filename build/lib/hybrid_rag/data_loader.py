import os
import glob
from typing import List, Dict

# จำเป็นต้องมี langchain-text-splitters ติดตั้งอยู่
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, data_config: dict):
        self.docs_path = data_config['docs_path']
        self.chunk_size = data_config['chunk_size']
        self.chunk_overlap = data_config['chunk_overlap']
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def load_documents(self) -> List[Dict]:
        """Loads text and markdown files from the specified directory."""
        if not os.path.exists(self.docs_path):
            print(f"Warning: Directory not found: {self.docs_path}")
            return []

        documents = []
        search_pattern = os.path.join(self.docs_path, "**", "*.*")
        
        for file_path in glob.glob(search_pattern, recursive=True):
            if file_path.endswith('.txt') or file_path.endswith('.md'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        documents.append({"source": file_path, "content": text})
                except Exception as e:
                    print(f"Warning: Failed to read {file_path}. Details: {e}")
                    
        return documents

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Splits documents into smaller chunks based on config."""
        chunked_data = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                chunked_data.append({
                    "source": doc["source"],
                    "chunk_id": i,
                    "content": chunk
                })
                
        return chunked_data

    def process(self) -> List[Dict]:
        """Executes the full pipeline: Load -> Chunk."""
        raw_docs = self.load_documents()
        if not raw_docs:
            print("Status: No valid documents found to process.")
            return []
            
        chunks = self.chunk_documents(raw_docs)
        return chunks