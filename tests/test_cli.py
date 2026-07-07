"""CLI tests via typer's CliRunner (deterministic paths only)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from substitutes_agent.cli import app

runner = CliRunner()


def test_cli_run_sample(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "out"
    result = runner.invoke(app, ["run", "--sample", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "Pipeline complete" in result.output
    assert (out / "step4_gap_report.md").exists()


def test_cli_consistency_skips_without_keys(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "out"
    runner.invoke(app, ["run", "--sample", "--output", str(out)])
    result = runner.invoke(app, ["consistency", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "Step 5 skipped" in result.output
    assert (out / "step5_consistency.json").exists()


def test_cli_consistency_fails_without_ontology(
    tmp_path: pytest.TempPathFactory,
) -> None:
    out = tmp_path / "empty"
    out.mkdir()
    result = runner.invoke(app, ["consistency", "--output", str(out)])
    assert result.exit_code == 2
    assert "run `substitutes-agent run` first" in result.output


def test_cli_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "consistency" in result.output
    assert "download" in result.output
