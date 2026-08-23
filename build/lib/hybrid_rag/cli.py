import typer
import yaml
import os
from pydantic import ValidationError
from .schema import RagConfig
from .env_diagnostics import check_hardware_and_env
from .data_loader import DocumentProcessor
from .vector_store import BaseVectorStore

app = typer.Typer(help="Hybrid RAG Builder CLI - Research Edition")

@app.command()
def init(force: bool = typer.Option(False, "--force", help="Overwrite existing configuration file")):
    """Generate a default config.yaml template."""
    if os.path.exists("config.yaml") and not force:
        typer.echo("Warning: config.yaml already exists. Use --force to overwrite.")
        raise typer.Exit(1)
    
    config_template = {
        "project_name": "research_rag_project",
        "mode": "offline",
        "offline": {
            "vector_db": "faiss",
            "llm_model": "ollama/llama3",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "api_key_env_var": None
        },
        "online": {
            "vector_db": "pinecone",
            "llm_model": "gemini-1.5-flash",
            "embedding_model": "text-embedding-004",
            "api_key_env_var": "GEMINI_API_KEY"
        },
        "data": {
            "docs_path": "./data",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    }
    
    with open("config.yaml", "w") as f:
        yaml.dump(config_template, f, sort_keys=False)
    typer.echo("System: config.yaml initialized successfully.")

@app.command()
def validate(config: str = typer.Option("config.yaml", help="Path to configuration file")):
    """Perform static schema validation on the configuration file."""
    if not os.path.exists(config):
         typer.echo(f"Error: Configuration file '{config}' not found.")
         raise typer.Exit(1)

    try:
        with open(config, "r") as f:
            raw_data = yaml.safe_load(f)
        
        valid_config = RagConfig(**raw_data)
        typer.echo(f"Validation: PASS. Project: {valid_config.project_name}")
    except ValidationError as e:
        typer.echo("Validation: FAIL. Schema errors detected:")
        typer.echo(e)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"System Error: Failed to parse YAML file. Details: {str(e)}")
        raise typer.Exit(1)

@app.command()
def diagnostics(config: str = typer.Option("config.yaml")):
    """Execute system hardware and environment variable probing."""
    if not os.path.exists(config):
         typer.echo(f"Error: Configuration file '{config}' not found.")
         raise typer.Exit(1)

    with open(config, "r") as f:
        raw_data = yaml.safe_load(f)
        
    typer.echo("--- System Diagnostics Report ---")
    report = check_hardware_and_env(raw_data)
    
    typer.echo(f"Python Runtime   : {report['python_version']}")
    typer.echo(f"Compute Backend  : {report['compute_device']}")
    
    if report['missing_api_keys']:
        typer.echo(f"Warning: Missing required environment variables: {report['missing_api_keys']}")
    else:
        typer.echo("Environment Setup: PASS")
    typer.echo("---------------------------------")

@app.command()
def ingest(config: str = typer.Option("config.yaml", help="Path to configuration file")):
    """Process documents and build the vector database."""
    if not os.path.exists(config):
         typer.echo(f"Error: Configuration file '{config}' not found.")
         raise typer.Exit(1)

    with open(config, "r") as f:
        raw_data = yaml.safe_load(f)
        
    mode = raw_data['mode']
    typer.echo(f"--- Starting Data Ingestion Pipeline (Mode: {mode.upper()}) ---")
    
    # 1. Process Documents
    processor = DocumentProcessor(raw_data['data'])
    typer.echo(f"Scanning directory: {raw_data['data']['docs_path']}")
    
    chunks = processor.process()
    typer.echo(f"Total chunks created: {len(chunks)}")
    
    if not chunks:
        typer.echo("Ingestion aborted: No data.")
        raise typer.Exit(1)

    # 2. Build Vector Index
    typer.echo("Building Vector Database...")
    vstore = BaseVectorStore(raw_data, mode)
    vstore.build_index(chunks)
    
    typer.echo("--- Data Ingestion Pipeline: PASS ---")
    
if __name__ == "__main__":
    app()