from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

class DataConfig(BaseModel):
    docs_path: str = Field(description="Directory path containing documents")
    chunk_size: int = Field(gt=0, le=4096, description="Text chunk size (1-4096)")
    chunk_overlap: int = Field(ge=0, description="Overlap token count between chunks")

    @field_validator('chunk_overlap')
    def check_overlap(cls, v, info):
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        return v

class LLMConfig(BaseModel):
    vector_db: Literal["chroma", "faiss", "pinecone", "qdrant"]
    llm_model: str
    embedding_model: str
    api_key_env_var: Optional[str] = None

class RagConfig(BaseModel):
    project_name: str
    mode: Literal["offline", "online"]
    offline: LLMConfig
    online: LLMConfig
    data: DataConfig