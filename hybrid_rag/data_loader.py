import os
import glob
import logging
from typing import List, Dict
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
import pandas as pd
import docx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Loads and chunks documents with comprehensive error handling and contextual metadata."""
    
    def __init__(self, data_config: dict):
        self.docs_path = data_config['docs_path']
        self.chunk_size = data_config['chunk_size']
        self.chunk_overlap = data_config['chunk_overlap']
        self.urls = data_config.get('urls', [])

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap must be less than chunk_size")
        
        logger.info(
            f"DocumentProcessor initialized: docs_path={self.docs_path}, "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )
        
        # ✅ SEMANTIC SPLITTER: กำหนด Separators ให้เน้นหั่นตาม "ย่อหน้า (\n\n)" หรือ "ประโยค (. )" 
        # เพื่อไม่ให้ความหมายขาดตอน (หลีกเลี่ยงการหั่นกลางประโยค)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        self.stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_failed": 0,
            "total_chunks": 0,
            "empty_chunks_skipped": 0,
        }

    # ---------------------------------------------------------
    # MULTI-FORMAT READERS
    # ---------------------------------------------------------
    def _read_pdf(self, file_path: str) -> str:
        text_parts = []
        try:
            with pymupdf.open(file_path) as doc:
                for page in doc:
                    text_parts.append(page.get_text())
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Failed to extract PDF {file_path}: {e}")
            raise

    def _read_csv(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path).dropna(how='all')
            return "\n".join([f"Row {idx}: " + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)]) for idx, row in df.iterrows()])
        except Exception as e:
            logger.error(f"Failed to extract CSV {file_path}: {e}")
            raise

    def _read_excel(self, file_path: str) -> str:
        try:
            df = pd.read_excel(file_path).dropna(how='all')
            return "\n".join([f"Row {idx}: " + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)]) for idx, row in df.iterrows()])
        except Exception as e:
            logger.error(f"Failed to extract Excel {file_path}: {e}")
            raise

    def _read_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            logger.error(f"Failed to extract DOCX {file_path}: {e}")
            raise

    def _read_html(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                return soup.get_text(separator='\n\n', strip=True)
        except Exception as e:
            logger.error(f"Failed to extract HTML {file_path}: {e}")
            raise

    def _read_url(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text(separator='\n\n', strip=True)
        except Exception as e:
            logger.error(f"Failed to extract URL {url}: {e}")
            raise



    # ---------------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------------
    def load_documents(self) -> List[Dict]:
        documents = []
        
        # ==========================================
        # 1. โหลดข้อมูลจากไฟล์ในโฟลเดอร์ (Local Files)
        # ==========================================
        if os.path.exists(self.docs_path) and os.path.isdir(self.docs_path):
            search_pattern = os.path.join(self.docs_path, "**", "*.*")
            file_paths = list(glob.glob(search_pattern, recursive=True))
            supported_exts = ['.txt', '.md', '.pdf', '.csv', '.xlsx', '.docx', '.html', '.htm']
            
            for file_path in file_paths:
                ext = os.path.splitext(file_path)[1].lower()
                if ext not in supported_exts:
                    continue
                
                self.stats["files_found"] += 1
                if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
                    self.stats["files_failed"] += 1
                    continue
                
                try:
                    text = ""
                    if ext in ['.txt', '.md']:
                        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
                    elif ext == '.pdf': text = self._read_pdf(file_path)
                    elif ext == '.csv': text = self._read_csv(file_path)
                    elif ext == '.xlsx': text = self._read_excel(file_path)
                    elif ext == '.docx': text = self._read_docx(file_path)
                    elif ext in ['.html', '.htm']: text = self._read_html(file_path)
                    
                    if not text.strip():
                        self.stats["files_failed"] += 1
                        continue
                    
                    file_title = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title()
                    documents.append({
                        "source": file_path,
                        "title": file_title,
                        "content": text,
                        "size_bytes": len(text.encode('utf-8'))
                    })
                    self.stats["files_processed"] += 1
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    self.stats["files_failed"] += 1

        # ==========================================
        # 2. ✅ NEW: โหลดข้อมูลจาก Web URLs
        # ==========================================
        if self.urls:
            logger.info(f"Fetching data from {len(self.urls)} URLs...")
            for url in self.urls:
                self.stats["files_found"] += 1 # นับ URL เป็นไฟล์นึง
                try:
                    text = self._read_url(url)
                    if not text.strip():
                        self.stats["files_failed"] += 1
                        continue
                    
                    documents.append({
                        "source": url,
                        "title": url.split('//')[-1].split('/')[0], # ใช้ Domain Name เป็น Title คร่าวๆ
                        "content": text,
                        "size_bytes": len(text.encode('utf-8'))
                    })
                    self.stats["files_processed"] += 1
                    logger.debug(f"Loaded URL: {url}")
                except Exception as e:
                    logger.warning(f"Failed to fetch {url}: {e}")
                    self.stats["files_failed"] += 1

        return documents
    

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        if not documents: return []
        
        chunked_data = []
        for doc in documents:
            content = doc.get("content", "").strip()
            source = doc.get("source", "unknown")
            title = doc.get("title", "Unknown Document")
            
            try:
                chunks = self.text_splitter.split_text(content)
                valid_chunks = []
                
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.strip()
                    if len(chunk_text.split()) < 3:
                        self.stats["empty_chunks_skipped"] += 1
                        continue
                    
                    # ✅ CONTEXTUAL CHUNKING: แปะชื่อเอกสารนำหน้าทุกๆ Chunk
                    # เพื่อให้ Vector DB และ AI ไม่ลืมว่าข้อความนี้มาจากเอกสารอะไร
                    enriched_content = f"[Document Title: {title}]\n{chunk_text}"
                    
                    chunked_data.append({
                        "source": source,
                        "chunk_id": i,
                        "content": enriched_content,
                        "chunk_size": len(enriched_content)
                    })
                    valid_chunks.append(enriched_content)
                
                self.stats["total_chunks"] += len(valid_chunks)
                
            except Exception as e:
                logger.error(f"Error chunking {source}: {e}")
                self.stats["files_failed"] += 1
                
        return chunked_data

    def process(self) -> List[Dict]:
        logger.info("=== Starting Document Processing Pipeline ===")
        raw_docs = self.load_documents()
        chunks = self.chunk_documents(raw_docs)
        
        logger.info("=== Document Processing Complete ===")
        logger.info(f"Files found:          {self.stats['files_found']}")
        logger.info(f"Files processed:      {self.stats['files_processed']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        
        return chunks
    
    def get_stats(self) -> Dict:
        return self.stats.copy()