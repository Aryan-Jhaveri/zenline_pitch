"""Pipeline orchestrator: runs steps 1-4 in order, writes all artifacts.

Step 5 (consistency experiment) is a separate CLI subcommand because it
requires LLM keys and is not part of the deterministic run.

Every step writes its artifact to output/ and appends a RunLogEntry to
output/run_log.json. Same input -> same output.
"""

from __future__ import annotations

import json
from pathlib import Path

from substitutes_agent.models import RunLogEntry
from substitutes_agent.step1_ingest import ingest
from substitutes_agent.step2_ontology import build_ontology
from substitutes_agent.step3_classify import classify
from substitutes_agent.step4_report import build_report

DEFAULT_INPUT = "data/styles.csv"
SAMPLE_INPUT = "data/sample.parquet"


def _write_run_log(path: Path, entries: list[RunLogEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.model_dump() for e in entries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_pipeline(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = "output",
    sample: bool = False,
) -> list[RunLogEntry]:
    """Run steps 1-4 and write all artifacts under output_dir."""
    if sample:
        input_path = SAMPLE_INPUT
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / ".llm_cache"
    run_log_path = out / "run_log.json"

    entries: list[RunLogEntry] = []

    # Step 1 — ingest & normalize.
    df, e1 = ingest(input_path)
    step1_path = out / "step1_normalized.parquet"
    df.write_parquet(step1_path)
    entries.append(e1)

    # Step 2 — build ontology.
    step2_path = out / "step2_ontology.json"
    _, e2 = build_ontology(step1_path, step2_path, cache_dir=cache_dir)
    entries.append(e2)

    # Write run log so far so Step 4 can read wall-clock for steps 1-3.
    _write_run_log(run_log_path, entries)

    # Step 3 — classify relationships.
    step3_path = out / "step3_relationships.json"
    decisions_path = out / "step3_llm_decisions.jsonl"
    _, e3 = classify(step2_path, step3_path, decisions_path=decisions_path)
    entries.append(e3)
    _write_run_log(run_log_path, entries)

    # Step 4 — gap report (reads run_log.json for wall-clock).
    step4_json = out / "step4_gap_report.json"
    step4_md = out / "step4_gap_report.md"
    _, e4 = build_report(step2_path, step3_path, run_log_path, step4_json, step4_md)
    entries.append(e4)
    _write_run_log(run_log_path, entries)

    return entries
