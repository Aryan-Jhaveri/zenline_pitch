"""End-to-end determinism: stable artifacts across two runs on the sample.

The core data artifacts (step1 normalized, step2 ontology, step3
relationships) must be byte-identical across runs. step4 and run_log
intentionally carry wall-clock durations and timestamps (per spec), so
they are compared with the time-varying fields stripped out.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from substitutes_agent import pipeline

BYTE_IDENTICAL_ARTIFACTS = (
    "step1_normalized.parquet",
    "step2_ontology.json",
    "step3_relationships.json",
)


def _hash_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _step4_without_wall_clock(p: Path) -> dict[str, object]:
    data = json.loads(p.read_text())
    data.pop("wall_clock_s", None)
    return data


def test_pipeline_e2e_deterministic(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    pipeline.run_pipeline(sample=True, output_dir=out1)
    pipeline.run_pipeline(sample=True, output_dir=out2)

    # Core data artifacts: byte-identical.
    for name in BYTE_IDENTICAL_ARTIFACTS:
        assert _hash_file(out1 / name) == _hash_file(out2 / name), (
            f"non-deterministic artifact: {name}"
        )

    # step4: identical once wall_clock_s (time-varying) is stripped.
    s4_1 = _step4_without_wall_clock(out1 / "step4_gap_report.json")
    s4_2 = _step4_without_wall_clock(out2 / "step4_gap_report.json")
    assert s4_1 == s4_2, "step4 gap numbers are non-deterministic"


def test_pipeline_e2e_produces_all_artifacts(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "out"
    entries = pipeline.run_pipeline(sample=True, output_dir=out)
    assert len(entries) == 4
    for name in (
        "step1_normalized.parquet",
        "step2_ontology.json",
        "step3_relationships.json",
        "step4_gap_report.json",
        "step4_gap_report.md",
        "run_log.json",
    ):
        assert (out / name).exists()


def test_pipeline_e2e_run_log_has_four_steps(
    tmp_path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "out"
    pipeline.run_pipeline(sample=True, output_dir=out)
    log = json.loads((out / "run_log.json").read_text())
    steps = [e["step"] for e in log]
    assert steps == [
        "step1_ingest",
        "step2_ontology",
        "step3_classify",
        "step4_report",
    ]


def test_pipeline_e2e_no_llm_cache_without_keys(
    tmp_path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "out"
    pipeline.run_pipeline(sample=True, output_dir=out)
    cache = out / ".llm_cache"
    assert not cache.exists() or not list(cache.iterdir())
