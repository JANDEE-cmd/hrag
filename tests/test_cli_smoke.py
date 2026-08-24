"""
Comprehensive smoke tests for the `hrag` CLI.

These don't call any real LLM/embedding provider or load heavy models --
they check that all core CLI commands (init, validate, diagnostics, version, 
clear-index, --version) work end-to-end without crashing. Run with:

    pip install -e ".[test]"
    pytest

Each test runs inside an isolated tmp_path so nothing touches your real
config.yaml / vector_index.bin.
"""
import json
import os

import pytest
from typer.testing import CliRunner

from hybrid_rag.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run every test inside a throwaway directory."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.stdout.lower()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_init_creates_config(isolated_cwd):
    result = runner.invoke(app, ["init", "--template", "offline"])
    assert result.exit_code == 0

    config_path = isolated_cwd / "config.yaml"
    assert config_path.exists()

    import yaml
    with open(config_path) as f:
        data = yaml.safe_load(f)
    assert data["mode"] == "offline"
    assert "offline" in data and "online" in data
    assert "data" in data


def test_init_refuses_overwrite_without_force(isolated_cwd):
    (isolated_cwd / "config.yaml").write_text("mode: offline\n")

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1

    # Original content must be untouched.
    assert (isolated_cwd / "config.yaml").read_text() == "mode: offline\n"


def test_init_force_overwrites(isolated_cwd):
    (isolated_cwd / "config.yaml").write_text("mode: offline\n")

    result = runner.invoke(app, ["init", "--force", "--template", "online"])
    assert result.exit_code == 0

    import yaml
    with open(isolated_cwd / "config.yaml") as f:
        data = yaml.safe_load(f)
    assert data["mode"] == "online"


def test_validate_passes_on_valid_offline_config(isolated_cwd):
    runner.invoke(app, ["init", "--template", "offline"])

    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0
    assert "pass" in result.stdout.lower()


def test_validate_fails_on_missing_required_field(isolated_cwd):
    (isolated_cwd / "config.yaml").write_text("project_name: broken\n")

    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 1


def test_validate_json_output_is_parseable(isolated_cwd):
    runner.invoke(app, ["init", "--template", "offline"])

    result = runner.invoke(app, ["validate", "--output-format", "json"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"


def test_diagnostics_json_has_expected_keys(isolated_cwd):
    runner.invoke(app, ["init", "--template", "offline"])

    result = runner.invoke(app, ["diagnostics", "--output-format", "json"])
    assert result.exit_code == 0

    report = json.loads(result.stdout)
    assert "python_version" in report
    assert "compute_device" in report
    assert "missing_api_keys" in report


def test_diagnostics_flags_missing_online_api_key(isolated_cwd, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    runner.invoke(app, ["init", "--template", "online", "--force"])

    result = runner.invoke(app, ["diagnostics", "--output-format", "json"])
    assert result.exit_code == 0
    
    report = json.loads(result.stdout)
    assert "GEMINI_API_KEY" in report["missing_api_keys"]


def test_clear_index_removes_files(isolated_cwd):
    # Simulate existing index and metadata files
    (isolated_cwd / "vector_index.bin").write_text("fake index data")
    (isolated_cwd / "vector_metadata.json").write_text("[]")

    assert (isolated_cwd / "vector_index.bin").exists()
    assert (isolated_cwd / "vector_metadata.json").exists()

    result = runner.invoke(app, ["clear-index"])
    assert result.exit_code == 0

    assert not (isolated_cwd / "vector_index.bin").exists()
    assert not (isolated_cwd / "vector_metadata.json").exists()