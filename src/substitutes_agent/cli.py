"""Typer CLI entry point.

Subcommands:
  substitutes-agent download        fetch styles.csv via Kaggle
  substitutes-agent run             run steps 1-4 on the full dataset
  substitutes-agent run --sample    run on the committed 500-row fixture
  substitutes-agent consistency     run Step 5 (skips cleanly without keys)
  substitutes-agent consistency --dry-run   print the call estimate and abort
  substitutes-agent audit           generate Step 6 audit trail + vendor diff
"""

from __future__ import annotations

from pathlib import Path

import typer

from substitutes_agent import pipeline
from substitutes_agent.step5_consistency import run_consistency
from substitutes_agent.step6_audit import run_step6

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def download() -> None:
    """Fetch the apparel styles.csv via the Kaggle CLI."""
    from data.download import download as _download

    _download()


@app.command()
def run(
    sample: bool = typer.Option(
        False, "--sample", help="Run on the committed data/sample.parquet fixture."
    ),
    input: str = typer.Option(
        pipeline.DEFAULT_INPUT, "--input", help="Input styles.csv / parquet path."
    ),
    output: str = typer.Option("output", "--output", help="Output directory."),
) -> None:
    """Run steps 1-4 (Ingest -> Ontology -> Classify -> Gap Report)."""
    entries = pipeline.run_pipeline(input_path=input, output_dir=output, sample=sample)
    typer.echo(f"Pipeline complete: {len(entries)} steps.")
    for e in entries:
        typer.echo(f"  {e.step}: {e.output_rows} rows in {e.duration_s}s")
    out = Path(output)
    typer.echo(f"Artifacts written under {out.resolve()}")
    typer.echo("  done")


@app.command()
def consistency(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the call estimate and abort."
    ),
    output: str = typer.Option("output", "--output", help="Output directory."),
) -> None:
    """Run Step 5: consistency experiment on borderline pairs."""
    out = Path(output)
    ontology_path = out / "step2_ontology.json"
    if not ontology_path.exists():
        typer.echo(
            f"error: run `substitutes-agent run` first; {ontology_path} not found.",
            err=True,
        )
        raise typer.Exit(2)
    report, log = run_consistency(
        ontology_path,
        output_json=out / "step5_consistency.json",
        output_md=out / "step5_consistency.md",
        transcripts_path=out / "step5_transcripts.jsonl",
        dry_run=dry_run,
    )
    if report.get("skipped"):
        typer.echo(f"Step 5 skipped: {report['reason']}")
    else:
        typer.echo(
            f"Step 5: cross-model agreement "
            f"{log.filters.get('cross_model_agreement_pct', 0)}%"
        )
    typer.echo("  done")


@app.command()
def audit(
    output: str = typer.Option(
        "output", "--output", help="Output directory with step 5 artifacts."
    ),
) -> None:
    """Generate step 6 audit trail and vendor-swap diff from step 5 artifacts."""
    out = Path(output)
    ontology_path = out / "step2_ontology.json"
    transcripts_path = out / "step5_transcripts.jsonl"
    consistency_json_path = out / "step5_consistency.json"
    missing = [
        p
        for p in (ontology_path, transcripts_path, consistency_json_path)
        if not p.exists()
    ]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        typer.echo(
            f"error: run `substitutes-agent consistency` first; missing: {joined}.",
            err=True,
        )
        raise typer.Exit(2)
    log = run_step6(
        ontology_path=ontology_path,
        transcripts_path=transcripts_path,
        consistency_json_path=consistency_json_path,
        audit_md_out=out / "step6_audit_trail.md",
        vendor_diff_md_out=out / "step6_vendor_diff.md",
        vendor_diff_json_out=out / "step6_vendor_diff.json",
    )
    typer.echo(
        f"Step 6: {log.filters.get('vendor_split_count', 0)} vendor-split "
        f"pair(s) of {log.filters.get('total_pairs', 0)} total pairs."
    )
    typer.echo("  done")


if __name__ == "__main__":
    app()
