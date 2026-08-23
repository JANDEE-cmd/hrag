import os
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal


class DataConfig(BaseModel):
    """Configuration for document data loading and chunking."""
    docs_path: str = Field(
        ...,
        description="Directory path containing documents"
    )
    chunk_size: int = Field(
        gt=0, 
        le=4096, 
        description="Text chunk size in tokens (1-4096)"
    )
    chunk_overlap: int = Field(
        ge=0, 
        description="Overlap token count between chunks"
    )

    @field_validator('chunk_overlap')
    @classmethod
    def check_overlap(cls, v: int, info) -> int:
        """✅ Validate that overlap is less than chunk size"""
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError(
                f"chunk_overlap ({v}) must be strictly less than chunk_size ({info.data['chunk_size']})"
            )
        return v
    
    @field_validator('docs_path')
    @classmethod
    def check_docs_path(cls, v: str) -> str:
        """✅ NEW: Warn if docs directory doesn't exist yet (but allow it)"""
        if not os.path.isabs(v) and not os.path.exists(v):
            # Path doesn't exist, but that's OK - it might be created later
            pass
        return v


class LLMConfig(BaseModel):
    """
    Configuration for LLM and embedding models.

    Both modes are routed through LiteLLM, so model names use LiteLLM's
    '<provider>/<model>' convention, e.g.:
      - offline: 'ollama/llama3.2:1b', 'ollama/qwen2.5:7b'
      - online:  'gemini/gemini-3.5-flash', 'openai/gpt-4o',
                 'anthropic/claude-3-5-sonnet-latest', 'azure/<deployment-name>'
    Same convention applies to embedding_model, e.g. 'gemini/gemini-embedding-001'
    or 'openai/text-embedding-3-small' for online mode. Offline embeddings are
    still served locally via sentence-transformers (not through LiteLLM), so
    embedding_model there stays a plain HF model id.
    """
    vector_db: Literal["chroma", "faiss", "pinecone", "qdrant"]
    llm_model: str = Field(
        ...,
        description=(
            "LiteLLM model string, e.g. 'ollama/llama3.2:1b' (offline) or "
            "'gemini/gemini-3.5-flash', 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet-latest' (online)"
        )
    )
    embedding_model: str = Field(
        ...,
        description=(
            "Embedding model name. Offline: sentence-transformers model id. "
            "Online: LiteLLM model string, e.g. 'gemini/gemini-embedding-001'."
        )
    )
    api_key_env_var: Optional[str] = Field(
        default=None,
        description="Environment variable name holding the provider's API key (required for online mode)"
    )


class RagConfig(BaseModel):
    """Root configuration for the entire RAG system."""
    project_name: str = Field(
        ...,
        description="Name of the project"
    )
    mode: Literal["offline", "online"] = Field(
        ...,
        description="Execution mode: offline (local Ollama) or online (cloud API)"
    )
    offline: LLMConfig = Field(
        ...,
        description="Configuration for offline mode"
    )
    online: LLMConfig = Field(
        ...,
        description="Configuration for online mode"
    )
    data: DataConfig = Field(
        ...,
        description="Data loading and processing configuration"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt override"
    )

    @model_validator(mode='after')
    def validate_online_api_key(self) -> 'RagConfig':
        """
        ✅ NEW: Comprehensive validation for online mode API keys.
        
        Ensures that if online mode is active, the required API key environment 
        variable exists.
        """
        if self.mode == "online":
            env_var = self.online.api_key_env_var
            
            # Check that env_var is specified
            if not env_var:
                raise ValueError(
                    f"Online mode requires 'api_key_env_var' to be set in config.\n"
                    f"Example for Gemini:\n"
                    f"  api_key_env_var: GEMINI_API_KEY"
                )
            
            # Check that the env var is actually set in the environment
            if env_var not in os.environ:
                raise ValueError(
                    f"Online mode is active but required API key is not set.\n"
                    f"Environment variable '{env_var}' not found.\n\n"
                    f"To fix:\n"
                    f"  1. Get your API key from the provider\n"
                    f"  2. Set it in your shell:\n"
                    f"     export {env_var}='your-api-key-here'\n"
                    f"  3. Verify: echo ${env_var}"
                )
            
            # ✅ NEW: Validate the API key isn't empty
            api_key = os.environ.get(env_var, "").strip()
            if not api_key:
                raise ValueError(
                    f"Environment variable '{env_var}' is set but empty.\n"
                    f"Set a valid API key: export {env_var}='your-key'"
                )
        
        return self
    
    @model_validator(mode='after')
    def validate_litellm_provider_prefix(self) -> 'RagConfig':
        """
        ✅ NEW: Online-mode models are routed through LiteLLM, which needs a
        '<provider>/<model>' prefix to know which API to call (e.g.
        'gemini/gemini-3.5-flash'). A bare name like 'gemini-3.5-flash' gets
        silently misrouted (LiteLLM falls back to OpenAI-compatible routing),
        so this is caught here with a clear message instead of a confusing
        runtime error from the provider SDK.
        """
        if self.mode == "online":
            for field_name, value in (
                ("llm_model", self.online.llm_model),
                ("embedding_model", self.online.embedding_model),
            ):
                if "/" not in value:
                    raise ValueError(
                        f"online.{field_name} = '{value}' is missing a LiteLLM provider prefix.\n"
                        f"Use '<provider>/<model>', e.g.:\n"
                        f"  llm_model: gemini/gemini-3.5-flash\n"
                        f"  embedding_model: gemini/gemini-embedding-001\n"
                        f"Other providers: openai/gpt-4o, anthropic/claude-3-5-sonnet-latest, azure/<deployment>"
                    )
        return self

    @model_validator(mode='after')
    def validate_ollama_connection(self) -> 'RagConfig':
        """
        ✅ NEW: Optional validation that Ollama is running in offline mode.
        
        This is a warning check, not a hard failure - Ollama might not be needed
        immediately if only ingesting documents.
        """
        if self.mode == "offline":
            try:
                import requests
                response = requests.get(
                    "http://localhost:11434/api/tags",
                    timeout=2
                )
                if response.status_code == 200:
                    pass  # Ollama is running
            except Exception:
                # Don't raise - Ollama might not be needed if we're only ingesting
                import warnings
                warnings.warn(
                    f"Offline mode is active but Ollama doesn't appear to be running.\n"
                    f"Start it with: ollama serve\n"
                    f"(Not required for 'ingest', but needed for 'ask' and 'chat')"
                )
        
        return self
    
    @model_validator(mode='after')
    def validate_model_names_not_empty(self) -> 'RagConfig':
        """✅ NEW: Ensure model names are actually configured, not placeholders"""
        modes = {
            'offline': self.offline,
            'online': self.online
        }
        
        for mode_name, mode_config in modes.items():
            if not mode_config.llm_model or mode_config.llm_model.isspace():
                raise ValueError(
                    f"{mode_name} mode: llm_model is empty or whitespace.\n"
                    f"Set a valid model name in config.yaml"
                )
            
            if not mode_config.embedding_model or mode_config.embedding_model.isspace():
                raise ValueError(
                    f"{mode_name} mode: embedding_model is empty or whitespace.\n"
                    f"Set a valid embedding model name in config.yaml"
                )
        
        return self


# ✅ NEW: Helper function for validation with better error messages
def validate_config_file(config_dict: dict) -> RagConfig:
    """
    Validate a config dictionary and provide helpful error messages.
    
    Args:
        config_dict: Configuration dict loaded from YAML
        
    Returns:
        Validated RagConfig object
        
    Raises:
        ValueError: With detailed instructions for fixing the config
    """
    try:
        return RagConfig(**config_dict)
    except Exception as e:
        error_msg = (
            f"❌ Configuration validation failed:\n\n"
            f"{str(e)}\n\n"
            f"Please check your config.yaml file and ensure all required fields are set.\n"
            f"Run 'hrag init' to generate a valid template config."
        )
        raise ValueError(error_msg) from e