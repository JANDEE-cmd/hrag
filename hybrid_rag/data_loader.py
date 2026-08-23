import os
import glob
import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Loads and chunks documents with comprehensive error handling."""
    
    def __init__(self, data_config: dict):
        self.docs_path = data_config['docs_path']
        self.chunk_size = data_config['chunk_size']
        self.chunk_overlap = data_config['chunk_overlap']
        
        # ✅ NEW: Validate config values
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        logger.info(
            f"DocumentProcessor initialized: docs_path={self.docs_path}, "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        # ✅ NEW: Track statistics
        self.stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_failed": 0,
            "total_chunks": 0,
            "empty_chunks_skipped": 0,
        }

    def load_documents(self) -> List[Dict]:
        """
        Loads text and markdown files from the specified directory.
        
        ✅ FIXED: Better error handling and file validation
        """
        if not os.path.exists(self.docs_path):
            logger.warning(f"Directory not found: {self.docs_path}")
            return []

        if not os.path.isdir(self.docs_path):
            raise ValueError(f"Not a directory: {self.docs_path}")

        documents = []
        search_pattern = os.path.join(self.docs_path, "**", "*.*")
        
        file_paths = list(glob.glob(search_pattern, recursive=True))
        logger.info(f"Found {len(file_paths)} files, filtering for .txt and .md...")
        
        for file_path in file_paths:
            # Only process text and markdown files
            if not (file_path.endswith('.txt') or file_path.endswith('.md')):
                continue
            
            self.stats["files_found"] += 1
            
            # ✅ NEW: Validate file is readable
            if not os.path.isfile(file_path):
                logger.warning(f"Skipping non-file: {file_path}")
                self.stats["files_failed"] += 1
                continue
            
            if not os.access(file_path, os.R_OK):
                logger.warning(f"File not readable: {file_path}")
                self.stats["files_failed"] += 1
                continue
            
            # ✅ NEW: Check file size (warn on very large files)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 50:
                logger.warning(f"Large file ({file_size_mb:.1f}MB): {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # ✅ NEW: Validate content isn't empty
                if not text or not text.strip():
                    logger.warning(f"Empty file: {file_path}")
                    self.stats["files_failed"] += 1
                    continue
                
                documents.append({
                    "source": file_path,
                    "content": text,
                    "size_bytes": len(text.encode('utf-8'))
                })
                self.stats["files_processed"] += 1
                logger.debug(f"Loaded: {file_path} ({len(text)} chars)")
                
            except UnicodeDecodeError as e:
                logger.warning(
                    f"Failed to read {file_path} (encoding error): {e}. "
                    f"Ensure file is UTF-8 encoded."
                )
                self.stats["files_failed"] += 1
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                self.stats["files_failed"] += 1
        
        logger.info(
            f"Loaded {self.stats['files_processed']} files "
            f"({self.stats['files_failed']} failed)"
        )
        return documents

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Splits documents into smaller chunks based on config.
        
        ✅ FIXED: Comprehensive input validation and error handling
        """
        if not documents:
            logger.warning("No documents to chunk")
            return []
        
        chunked_data = []
        
        for doc in documents:
            # ✅ NEW: Validate document structure
            content = doc.get("content", "").strip()
            source = doc.get("source", "unknown")
            
            if not content:
                logger.warning(f"Skipping document with empty content: {source}")
                self.stats["files_failed"] += 1
                continue
            
            try:
                chunks = self.text_splitter.split_text(content)
                
                if not chunks:
                    logger.warning(
                        f"No chunks produced from {source} "
                        f"(content might be too small for chunk_size={self.chunk_size})"
                    )
                    self.stats["files_failed"] += 1
                    continue
                
                # ✅ NEW: Filter out empty chunks and track them
                valid_chunks = []
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.strip()
                    
                    if not chunk_text:
                        self.stats["empty_chunks_skipped"] += 1
                        continue
                    
                    # ✅ NEW: Additional validation
                    if len(chunk_text.split()) < 3:  # Too short
                        logger.debug(f"Skipping very short chunk from {source}")
                        self.stats["empty_chunks_skipped"] += 1
                        continue
                    
                    chunked_data.append({
                        "source": source,
                        "chunk_id": i,
                        "content": chunk_text,
                        "chunk_size": len(chunk_text)
                    })
                    valid_chunks.append(chunk_text)
                
                self.stats["total_chunks"] += len(valid_chunks)
                logger.debug(
                    f"✓ {source}: {len(valid_chunks)} valid chunks "
                    f"({len(chunks) - len(valid_chunks)} empty skipped)"
                )
                
            except Exception as e:
                logger.error(f"Error chunking {source}: {e}")
                self.stats["files_failed"] += 1
                continue
        
        return chunked_data

    def process(self) -> List[Dict]:
        """
        Executes the full pipeline: Load -> Chunk.
        
        Returns statistics about the processing.
        """
        logger.info("=== Starting Document Processing Pipeline ===")
        
        # Load
        raw_docs = self.load_documents()
        if not raw_docs:
            logger.error("No valid documents found to process")
            return []
        
        logger.info(f"Loaded {len(raw_docs)} documents")
        
        # Chunk
        chunks = self.chunk_documents(raw_docs)
        
        # Log final statistics
        logger.info("=== Document Processing Complete ===")
        logger.info(f"Files found:          {self.stats['files_found']}")
        logger.info(f"Files processed:      {self.stats['files_processed']}")
        logger.info(f"Files failed:         {self.stats['files_failed']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        logger.info(f"Empty chunks skipped: {self.stats['empty_chunks_skipped']}")
        
        if self.stats['total_chunks'] == 0:
            logger.error("No valid chunks produced - check your documents")
            return []
        
        return chunks
    
    def get_stats(self) -> Dict:
        """Return processing statistics."""
        return self.stats.copy()