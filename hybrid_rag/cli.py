import typer
import yaml
import os
import json
import sys
from enum import Enum
from typing import Optional, List

from rich.console import Console
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

from pydantic import ValidationError
from .schema import RagConfig
from .env_diagnostics import check_hardware_and_env
from .data_loader import DocumentProcessor
from .vector_store import BaseVectorStore
from .chat_engine import ChatEngine

__version__ = "0.1.0"

app = typer.Typer(
    help="Hybrid RAG CLI Interface",
    no_args_is_help=True,
    add_completion=True,
)
console = Console()
err_console = Console(stderr=True)

custom_style = Style.from_dict({
    'prompt': 'ansicyan bold',
})


# --------------------------------------------------------------------------
# Shared enums / option definitions
# --------------------------------------------------------------------------

class OutputFormat(str, Enum):
    text = "text"
    json = "json"
    markdown = "markdown"


class LogLevel(str, Enum):
    quiet = "quiet"
    normal = "normal"
    verbose = "verbose"
    debug = "debug"


class State:
    verbose: bool = False
    quiet: bool = False
    no_color: bool = False


state = State()

CONFIG_OPTION = typer.Option(
    "config.yaml", "--config", "-c", envvar="RAG_CONFIG",
    help="Path to configuration file.",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def log(msg: str, level: str = "info"):
    """Respect global verbosity flags when echoing status/progress info."""
    if state.quiet and level != "error":
        return
    if level == "debug" and not state.verbose:
        return
    if level == "error":
        err_console.print(f"[bold red]{msg}[/bold red]")
    elif level == "warn":
        console.print(f"[bold yellow]{msg}[/bold yellow]")
    elif level == "debug":
        console.print(f"[dim]DEBUG: {msg}[/dim]")
    else:
        console.print(msg)


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        err_console.print(f"[bold red]Error:[/bold red] Configuration file '{path}' not found.")
        raise typer.Exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_or_exit(raw_data: dict) -> RagConfig:
    """
    Run the full RagConfig schema check before raw_data is used to build any
    engine. This is what catches things like a missing LiteLLM provider
    prefix ('gemini-3.5-flash' instead of 'gemini/gemini-3.5-flash') up
    front with an actionable message, instead of letting it reach LiteLLM /
    the provider SDK and fail with a confusing low-level stack trace.

    Every command that loads config.yaml and hands it to ChatEngine,
    DocumentProcessor, or BaseVectorStore MUST call this after applying any
    CLI overrides -- 'validate' is not the only place this check runs.
    """
    try:
        RagConfig(**raw_data)
        return raw_data
    except ValidationError as e:
        err_console.print("[bold red]Configuration Error:[/bold red] config.yaml failed validation.")
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            err_console.print(f"  [red]\u2022[/red] {loc}: {err['msg']}")
        err_console.print(
            "\nRun '[bold]hrag validate[/bold]' for full details, "
            "or fix config.yaml and try again."
        )
        raise typer.Exit(1)


def apply_overrides(
    raw_data: dict,
    mode: Optional[str] = None,
    model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    vector_db: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> dict:
    """Apply CLI-level overrides on top of the loaded YAML config."""
    if mode:
        raw_data["mode"] = mode
    active_mode = raw_data["mode"]
    section = raw_data.setdefault(active_mode, {})
    if model:
        section["llm_model"] = model
    if embedding_model:
        section["embedding_model"] = embedding_model
    if vector_db:
        section["vector_db"] = vector_db
    if system_prompt:
        raw_data["system_prompt"] = system_prompt
    return raw_data


def emit(payload_text: str, output_format: OutputFormat, json_key: str = "response"):
    if output_format == OutputFormat.json:
        console.print_json(json.dumps({json_key: payload_text}))
    elif output_format == OutputFormat.markdown:
        console.print(Markdown(payload_text))
    else:
        console.print(payload_text)


# --------------------------------------------------------------------------
# Global callback (applies to every command, e.g. `rag --verbose ask ...`)
# --------------------------------------------------------------------------

def _version_callback(value: bool):
    if value:
        typer.echo(f"rag-cli version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose/debug logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-essential output."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored/rich output."),
    version: Optional[bool] = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show the CLI version and exit.",
    ),
):
    """Hybrid RAG CLI Interface."""
    state.verbose = verbose
    state.quiet = quiet and not verbose
    state.no_color = no_color
    if no_color:
        console.no_color = True
        err_console.no_color = True


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------

@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configuration file."),
    template: str = typer.Option(
        "offline", "--template", "-t",
        help="Starting template to seed as the active mode.",
        case_sensitive=False,
        show_choices=True,
        metavar="[offline|online]",
    ),
    output: str = typer.Option("config.yaml", "--output", "-o", help="Path to write the generated config."),
):
    """Generate a default config.yaml template."""
    if template not in ("offline", "online"):
        err_console.print("[bold red]Error:[/bold red] --template must be 'offline' or 'online'.")
        raise typer.Exit(1)

    if os.path.exists(output) and not force:
        log(f"Warning: {output} already exists. Use --force to overwrite.", "warn")
        raise typer.Exit(1)

    config_template = {
        "project_name": "research_rag_project",
        "mode": template,
        "offline": {
            "vector_db": "faiss",
            "llm_model": "ollama/llama3.2:1b",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "api_key_env_var": None
        },
        "online": {
            "vector_db": "faiss",
            # Models are routed through LiteLLM as "<provider>/<model>", so
            # swapping providers is just changing this string + the matching
            # api_key_env_var, e.g. "openai/gpt-4o" + OPENAI_API_KEY, or
            # "anthropic/claude-3-5-sonnet-latest" + ANTHROPIC_API_KEY.
            "llm_model": "gemini/gemini-3.5-flash",
            # NOTE: text-embedding-004 was deprecated by Google (returns 404
            # NOT_FOUND on embedContent as of 2026). gemini-embedding-001 is
            # the current replacement (3072-dim output, vs 768-dim previously).
            "embedding_model": "gemini/gemini-embedding-001",
            "api_key_env_var": "GEMINI_API_KEY"
        },
        "data": {
            "docs_path": "./data",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    }

    with open(output, "w") as f:
        yaml.dump(config_template, f, sort_keys=False)
    log(f"System: {output} initialized successfully (mode={template}).")


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

@app.command()
def validate(
    config: str = CONFIG_OPTION,
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as errors."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", "-o", case_sensitive=False,
        help="Output format for the validation result.",
    ),
):
    """Perform static schema validation on the configuration file."""
    raw_data = load_config(config)

    try:
        valid_config = RagConfig(**raw_data)
        if output_format == OutputFormat.json:
            console.print_json(json.dumps({"status": "pass", "project_name": valid_config.project_name}))
        else:
            log(f"Validation: PASS. Project: {valid_config.project_name}")
    except ValidationError as e:
        if output_format == OutputFormat.json:
            console.print_json(json.dumps({"status": "fail", "errors": json.loads(e.json())}))
        else:
            log("Validation: FAIL. Schema errors detected:", "error")
            typer.echo(e)
        raise typer.Exit(1)
    except Exception as e:
        err_console.print(f"[bold red]System Error:[/bold red] Failed to parse YAML file. Details: {str(e)}")
        raise typer.Exit(1)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

@app.command()
def diagnostics(
    config: str = CONFIG_OPTION,
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", "-o", case_sensitive=False,
        help="Output format for the diagnostics report.",
    ),
):
    """Execute system hardware and environment variable probing."""
    raw_data = load_config(config)
    report = check_hardware_and_env(raw_data)

    if output_format == OutputFormat.json:
        console.print_json(json.dumps(report))
        return

    log("--- System Diagnostics Report ---")
    log(f"Python Runtime   : {report['python_version']}")
    log(f"Compute Backend  : {report['compute_device']}")

    if report['missing_api_keys']:
        log(f"Warning: Missing required environment variables: {report['missing_api_keys']}", "warn")
    else:
        log("Environment Setup: PASS")
    log("---------------------------------")


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------

@app.command()
def ingest(
    config: str = CONFIG_OPTION,
    mode: Optional[str] = typer.Option(
        None, "--mode", help="Override the mode from config (offline|online) for this ingest run only.",
    ),
    force: bool = typer.Option(False, "--force", "--rebuild", help="Rebuild the index even if one already exists."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Scan and chunk documents without writing the index."),
    workers: int = typer.Option(1, "--workers", "-w", min=1, help="Parallel workers for document processing."),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Override chunk size from config."),
    chunk_overlap: Optional[int] = typer.Option(None, "--chunk-overlap", help="Override chunk overlap from config."),
    include: List[str] = typer.Option(None, "--include", help="Glob pattern(s) to include (repeatable)."),
    exclude: List[str] = typer.Option(None, "--exclude", help="Glob pattern(s) to exclude (repeatable)."),
):
    """Process documents and build the vector database."""
    raw_data = load_config(config)

    if mode:
        if mode not in ("offline", "online"):
            err_console.print("[bold red]Error:[/bold red] --mode must be 'offline' or 'online'.")
            raise typer.Exit(1)
        raw_data["mode"] = mode

    if chunk_size is not None:
        raw_data["data"]["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        raw_data["data"]["chunk_overlap"] = chunk_overlap
    if include:
        raw_data["data"]["include"] = include
    if exclude:
        raw_data["data"]["exclude"] = exclude

    raw_data = validate_or_exit(raw_data)

    active_mode = raw_data['mode']
    embedding_model = raw_data[active_mode].get('embedding_model', 'unknown')

    # An index built under one mode's embedding model is not compatible with
    # the other mode's embedder (different vector dimensions). Refuse to
    # silently overwrite an existing index when switching modes unless the
    # user explicitly confirms with --force.
    index_path = "vector_index.bin"
    metadata_path = "vector_metadata.json"
    existing_index = os.path.exists(index_path) or os.path.exists(metadata_path)
    if existing_index and not force and not dry_run:
        err_console.print(
            f"[bold red]Error:[/bold red] An existing vector index was found "
            f"({index_path}). Rebuilding under mode='{active_mode}' "
            f"(embedding_model='{embedding_model}') would replace it, and an index "
            f"built with a different embedding model is NOT compatible across modes "
            f"(different vector dimensions -> query errors).\n"
            f"Re-run with [bold]--force[/bold] to confirm the rebuild."
        )
        raise typer.Exit(1)

    log(f"--- Starting Data Ingestion Pipeline (Mode: {active_mode.upper()}, Embedding: {embedding_model}) ---")

    processor = DocumentProcessor(raw_data['data'])
    log(f"Scanning directory: {raw_data['data']['docs_path']}")

    chunks = processor.process()
    log(f"Total chunks created: {len(chunks)}")

    if not chunks:
        log("Ingestion aborted: No data.", "error")
        raise typer.Exit(1)

    if dry_run:
        log("Dry run: skipping index build.", "warn")
        log("--- Data Ingestion Pipeline: DRY-RUN COMPLETE ---")
        return

    log(f"Building Vector Database (force={force}, workers={workers})...")
    vstore = BaseVectorStore(raw_data, active_mode)
    vstore.build_index(chunks)

    log("--- Data Ingestion Pipeline: PASS ---")


# --------------------------------------------------------------------------
# ask
# --------------------------------------------------------------------------

@app.command()
def ask(
    query: str = typer.Argument(..., help="The question you want to ask."),
    config: str = CONFIG_OPTION,
    top_k: int = typer.Option(4, "--top-k", "-k", help="Number of context chunks to retrieve."),
    mode: Optional[str] = typer.Option(None, "--mode", help="Override the mode from config (offline|online)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the LLM model for this query."),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="Override the system prompt."),
    no_context: bool = typer.Option(False, "--no-context", help="Skip retrieval; query the model directly."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", "-o", case_sensitive=False,
        help="Output format for the response.",
    ),
    save: Optional[str] = typer.Option(None, "--save", help="Write the response to a file instead of/as well as stdout."),
):
    """Query the RAG system with a question."""
    raw_data = load_config(config)
    raw_data = apply_overrides(raw_data, mode=mode, model=model, system_prompt=system_prompt)
    raw_data = validate_or_exit(raw_data)
    active_mode = raw_data['mode']

    try:
        log(f"Initializing Chat Engine ({active_mode.upper()} Mode)...", "debug")
        engine = ChatEngine(raw_data, active_mode)

        context = "" if no_context else None
        if not no_context:
            log("Retrieving context from database...", "debug")
            context = engine.retrieve(query, top_k=top_k)

        log("Generating response...", "debug")
        response = engine.generate(query, context)

        if output_format == OutputFormat.text:
            typer.echo("\n================ RESPONSE ================\n")
        emit(response, output_format)
        if output_format == OutputFormat.text:
            typer.echo("\n==========================================\n")

        if save:
            with open(save, "w", encoding="utf-8") as f:
                f.write(response)
            log(f"Response saved to {save}")

    except Exception as e:
        err_console.print(f"[bold red]System Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------

@app.command()
def chat(
    config: str = CONFIG_OPTION,
    top_k: int = typer.Option(2, "--top-k", "-k", help="Number of context chunks to retrieve per turn."),
    mode: Optional[str] = typer.Option(None, "--mode", help="Override the mode from config (offline|online)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the LLM model for this session."),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="Override the system prompt."),
    max_turns: Optional[int] = typer.Option(None, "--max-turns", help="End the session automatically after N turns."),
    session_file: Optional[str] = typer.Option(
        None, "--resume", "--session", help="Load/save conversation history to this file."
    ),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable incremental/streaming output, if supported."),
):
    """Launches an interactive chat session."""
    raw_data = load_config(config)
    raw_data = apply_overrides(raw_data, mode=mode, model=model, system_prompt=system_prompt)
    raw_data = validate_or_exit(raw_data)
    active_mode = raw_data['mode']

    console.print(f"\n[bold cyan]Initializing Chat Engine [Mode: {active_mode.upper()}]...[/bold cyan]")
    try:
        engine = ChatEngine(raw_data, active_mode)
    except Exception as e:
        console.print(f"[bold red]Initialization Error: {e}[/bold red]")
        raise typer.Exit(1)

    history = []
    if session_file and os.path.exists(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            console.print(f"[dim]Resumed session with {len(history)} prior turn(s) from {session_file}.[/dim]")
        except Exception:
            console.print(f"[bold yellow]Warning: could not parse session file {session_file}; starting fresh.[/bold yellow]")

    console.print("[bold green]Session initialized successfully. Type your queries below. Type 'exit' to terminate.[/bold green]\n")
    console.print("-" * 60)

    session = PromptSession(style=custom_style)
    turn_count = 0

    while True:
        try:
            if max_turns is not None and turn_count >= max_turns:
                console.print(f"[bold yellow]Reached --max-turns={max_turns}. Ending session.[/bold yellow]")
                break

            user_input = session.prompt("\nUser > ")

            if user_input.strip().lower() in ["exit", "quit", "q"]:
                console.print("[bold yellow]Terminating session. Goodbye.[/bold yellow]")
                break

            if not user_input.strip():
                continue

            status_label = "Retrieving context and generating response..." if not no_stream else "Generating response..."
            with console.status(f"[bold cyan]{status_label}[/bold cyan]", spinner="dots"):
                context = engine.retrieve(user_input, top_k=top_k)
                response = engine.generate(user_input, context)

            console.print("\nAssistant:")
            console.print(Markdown(response))
            console.print("-" * 60)

            turn_count += 1
            history.append({"user": user_input, "assistant": response})
            if session_file:
                with open(session_file, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Session interrupted. Goodbye.[/bold yellow]")
            break


# --------------------------------------------------------------------------
# run  (single-command entry point: validate -> auto-ingest -> chat)
# --------------------------------------------------------------------------

@app.command()
def run(
    config: str = CONFIG_OPTION,
    top_k: int = typer.Option(2, "--top-k", "-k", help="Number of context chunks to retrieve per turn."),
    mode: Optional[str] = typer.Option(None, "--mode", help="Override the mode from config (offline|online)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the LLM model for this session."),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="Override the system prompt."),
    reingest: bool = typer.Option(False, "--reingest", help="Force a fresh ingest even if an index already exists."),
):
    """
    One-command RAG agent: validates config, builds the vector index if it
    doesn't exist yet (or --reingest is passed), then drops straight into
    an interactive chat session. This is the command most people want --
    'init' + 'ingest' + 'chat' without three separate steps.
    """
    raw_data = load_config(config)
    raw_data = apply_overrides(raw_data, mode=mode, model=model, system_prompt=system_prompt)
    raw_data = validate_or_exit(raw_data)
    active_mode = raw_data['mode']

    index_path = "vector_index.bin"
    metadata_path = "vector_metadata.json"
    has_index = os.path.exists(index_path) and os.path.exists(metadata_path)

    if not has_index or reingest:
        reason = "no existing index found" if not has_index else "--reingest requested"
        console.print(f"[bold cyan]Building knowledge base ({reason})...[/bold cyan]")

        processor = DocumentProcessor(raw_data['data'])
        log(f"Scanning directory: {raw_data['data']['docs_path']}")
        chunks = processor.process()

        if not chunks:
            err_console.print(
                f"[bold red]Error:[/bold red] No documents found under "
                f"'{raw_data['data']['docs_path']}'. Add some .txt/.md files there and try again."
            )
            raise typer.Exit(1)

        vstore = BaseVectorStore(raw_data, active_mode)
        vstore.build_index(chunks)
        console.print(f"[bold green]Knowledge base ready ({len(chunks)} chunks indexed).[/bold green]\n")
    else:
        console.print("[dim]Using existing vector index (run with --reingest to rebuild).[/dim]\n")

    console.print(f"[bold cyan]Initializing Chat Engine [Mode: {active_mode.upper()}]...[/bold cyan]")
    try:
        engine = ChatEngine(raw_data, active_mode)
    except Exception as e:
        console.print(f"[bold red]Initialization Error: {e}[/bold red]")
        raise typer.Exit(1)

    console.print("[bold green]Ready. Type your queries below. Type 'exit' to terminate.[/bold green]\n")
    console.print("-" * 60)

    session = PromptSession(style=custom_style)

    while True:
        try:
            user_input = session.prompt("\nUser > ")

            if user_input.strip().lower() in ["exit", "quit", "q"]:
                console.print("[bold yellow]Terminating session. Goodbye.[/bold yellow]")
                break
            if not user_input.strip():
                continue

            with console.status("[bold cyan]Retrieving context and generating response...[/bold cyan]", spinner="dots"):
                context = engine.retrieve(user_input, top_k=top_k)
                response = engine.generate(user_input, context)

            console.print("\nAssistant:")
            console.print(Markdown(response))
            console.print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Session interrupted. Goodbye.[/bold yellow]")
            break

if __name__ == "__main__":
    app()