"""Step 5 — consistency experiment on borderline pairs (optional).

Framing (kept in the report itself, not a comment): Arber's Q2 2026
newsletter argues that raw autonomous agents give inconsistent answers to
the same question. This step tests that claim on this project's own
ambiguous pairs — same prompt, same pair, repeated runs, multiple models.

Behavior:
  1. Take up to 30 candidate pairs (from Step 3's blocking+scoring) whose
     total_score fell in [0.40, 0.60]. Fewer -> use all; more ->
     deterministic sample with seed=0.
  2. For each pair, run each available model 3 times (same prompt as the
     Step 3 tie-breaker).
  3. Models: claude-haiku-4-5-20251001, claude-sonnet-4-6, gemini-2.5-flash.
     Skip any whose key isn't present. Zero keys -> skip Step 5 entirely.
  4. Per model: self-agreement rate (% of pairs where all 3 runs agreed).
     Cross-model: % of pairs where ALL models across ALL runs agreed.
     Baseline: rules-based Step 3 is 100% by construction.
  5. Emit step5_consistency.json + .md (with a placeholder paragraph to be
     filled after a real run) and step5_transcripts.jsonl.

Cost cap: 30 pairs * 3 models * 3 runs = 270 calls. --dry-run prints the
estimate and aborts. Live network paths are pragma: no cover.
"""

from __future__ import annotations

import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from substitutes_agent.llm import DEFAULT_MODELS, classify_pair, is_available
from substitutes_agent.models import OntologyRecord, RunLogEntry
from substitutes_agent.step3_classify import iter_candidate_pairs

SCORE_LOW = 0.40
SCORE_HIGH = 0.60
MAX_PAIRS = 30
RUNS_PER_MODEL = 3


def compute_consistency(
    verdicts: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    """Pure arithmetic over a verdicts structure.

    verdicts[pair_id][model] = list of "yes"/"no" verdicts (one per run).
    Returns per-model self-agreement and cross-model agreement rates.
    """
    pair_ids = list(verdicts.keys())
    n_pairs = len(pair_ids)
    if n_pairs == 0:
        return {
            "pairs": 0,
            "per_model_self_agreement": {},
            "cross_model_agreement_pct": 0.0,
            "rules_baseline_pct": 100.0,
        }

    models = sorted({m for p in verdicts.values() for m in p})
    per_model: dict[str, dict[str, float]] = {}
    for model in models:
        agreed = 0
        for pid in pair_ids:
            runs = verdicts[pid].get(model, [])
            if runs and len(set(runs)) == 1:
                agreed += 1
        per_model[model] = {
            "self_agreement_pct": round(100.0 * agreed / n_pairs, 2),
            "pairs_evaluated": n_pairs,
        }

    cross_agreed = 0
    for pid in pair_ids:
        all_verdicts: list[str] = []
        for model in models:
            all_verdicts.extend(verdicts[pid].get(model, []))
        if all_verdicts and len(set(all_verdicts)) == 1:
            cross_agreed += 1
    cross_pct = round(100.0 * cross_agreed / n_pairs, 2)

    return {
        "pairs": n_pairs,
        "per_model_self_agreement": per_model,
        "cross_model_agreement_pct": cross_pct,
        "rules_baseline_pct": 100.0,
    }


def _select_pairs(records: list[OntologyRecord]) -> list[tuple[str, str, float]]:
    """Candidate pairs whose total_score is in [0.40, 0.60], capped at 30."""
    pairs = iter_candidate_pairs(records)
    borderline = [
        (a.code, b.code, comp.total_score)
        for a, b, comp, _ in pairs
        if SCORE_LOW <= comp.total_score <= SCORE_HIGH
    ]
    borderline.sort(key=lambda t: (t[0], t[1]))
    if len(borderline) > MAX_PAIRS:
        rng = random.Random(0)
        idx = sorted(rng.sample(range(len(borderline)), MAX_PAIRS))
        borderline = [borderline[i] for i in idx]
    return borderline


def run_consistency(
    ontology_path: str | Path,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    transcripts_path: str | Path | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, Any], RunLogEntry]:
    """Run Step 5. Skips cleanly (no artifacts) when no provider key is set."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    records = [
        OntologyRecord(**d)
        for d in json.loads(Path(ontology_path).read_text(encoding="utf-8"))
    ]
    pairs = _select_pairs(records)

    available_models = [m for m in DEFAULT_MODELS if is_available(m)]
    estimated_calls = len(pairs) * len(available_models) * RUNS_PER_MODEL

    if not available_models:
        report: dict[str, Any] = {
            "skipped": True,
            "reason": "no LLM provider keys set; Step 5 requires at least one",
            "borderline_pairs_available": len(pairs),
            "models_evaluated": [],
            "estimated_calls": 0,
        }
        duration = time.monotonic() - started
        entry = RunLogEntry(
            step="step5_consistency",
            started_at=started_at,
            duration_s=round(duration, 4),
            input_rows=len(records),
            output_rows=0,
            filters={"borderline_pairs": len(pairs), "models": []},
            notes="Skipped: no provider keys.",
        )
        _write_outputs(report, output_json, output_md)
        return report, entry

    print(
        f"Step 5: {len(pairs)} pairs x {len(available_models)} models x "
        f"{RUNS_PER_MODEL} runs = {estimated_calls} calls (cap 270)."
    )
    if estimated_calls > 270:
        raise SystemExit(f"Step 5 cost cap exceeded: {estimated_calls} > 270 calls")
    if dry_run:
        report = {
            "skipped": True,
            "reason": "dry-run; printed estimate only",
            "borderline_pairs_available": len(pairs),
            "models_evaluated": available_models,
            "estimated_calls": estimated_calls,
        }
        duration = time.monotonic() - started
        entry = RunLogEntry(
            step="step5_consistency",
            started_at=started_at,
            duration_s=round(duration, 4),
            input_rows=len(records),
            output_rows=0,
            filters={
                "borderline_pairs": len(pairs),
                "models": available_models,
                "estimated_calls": estimated_calls,
            },
            notes="Dry run; no calls made.",
        )
        _write_outputs(report, output_json, output_md)
        return report, entry

    transcripts_fh = None
    if transcripts_path is not None:
        tp = Path(transcripts_path)
        tp.parent.mkdir(parents=True, exist_ok=True)
        transcripts_fh = open(tp, "a", encoding="utf-8")  # noqa: SIM115

    verdicts: dict[str, dict[str, list[str]]] = {}
    try:
        for sku_a, sku_b, score in pairs:  # pragma: no cover
            pid = f"{sku_a}|{sku_b}"
            verdicts[pid] = {}
            rec_a = next(r for r in records if r.code == sku_a)
            rec_b = next(r for r in records if r.code == sku_b)
            for model in available_models:  # pragma: no cover
                verdicts[pid][model] = []
                for _ in range(RUNS_PER_MODEL):
                    pv = classify_pair(model, rec_a.model_dump(), rec_b.model_dump())
                    verdicts[pid][model].append(pv.verdict)
                    if transcripts_fh is not None:
                        transcripts_fh.write(
                            json.dumps(
                                {
                                    "pair": pid,
                                    "model": model,
                                    "verdict": pv.verdict,
                                    "reason": pv.reason,
                                    "score": score,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
    finally:
        if transcripts_fh is not None:
            transcripts_fh.close()

    stats = compute_consistency(verdicts)
    report = {
        "skipped": False,
        "framing": (
            "Arber's Q2 2026 newsletter argues that raw autonomous agents "
            "give inconsistent answers to the same question. This step "
            "tests that claim on this project's own ambiguous pairs."
        ),
        "borderline_pairs": len(pairs),
        "models_evaluated": available_models,
        "estimated_calls": estimated_calls,
        "verdicts": verdicts,
        "stats": stats,
        "interpretation": "<fill in after seeing the numbers>",
    }

    duration = time.monotonic() - started
    entry = RunLogEntry(
        step="step5_consistency",
        started_at=started_at,
        duration_s=round(duration, 4),
        input_rows=len(records),
        output_rows=len(pairs),
        filters={
            "borderline_pairs": len(pairs),
            "models": available_models,
            "estimated_calls": estimated_calls,
            "cross_model_agreement_pct": stats["cross_model_agreement_pct"],
        },
        notes="Consistency experiment. Interpretation paragraph is a "
        "placeholder until a real run's numbers are reviewed.",
    )

    _write_outputs(report, output_json, output_md)

    return report, entry


def _write_outputs(
    report: dict[str, Any],
    output_json: str | Path | None,
    output_md: str | Path | None,
) -> None:
    """Write the JSON + Markdown artifacts for a (possibly skipped) run."""
    if output_json is not None:
        jp = Path(output_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if output_md is not None:
        mp = Path(output_md)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(_render_markdown(report), encoding="utf-8")


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Consistency Experiment")
    lines.append("")
    lines.append(report.get("framing", ""))
    lines.append("")
    if report.get("skipped"):
        lines.append(f"**Skipped:** {report['reason']}")
        lines.append("")
        lines.append(
            f"- Borderline pairs available: {report['borderline_pairs_available']}"
        )
        if report.get("models_evaluated"):
            lines.append(f"- Models: {', '.join(report['models_evaluated'])}")
        lines.append(f"- Estimated calls: {report['estimated_calls']}")
        return "\n".join(lines)

    stats = report["stats"]
    lines.append("## Within-model consistency")
    lines.append("")
    lines.append("| Model | Self-agreement % | Pairs |")
    lines.append("| --- | ---: | ---: |")
    for model, mstats in stats["per_model_self_agreement"].items():
        lines.append(
            f"| {model} | {mstats['self_agreement_pct']} | "
            f"{mstats['pairs_evaluated']} |"
        )
    lines.append("")
    lines.append("## Cross-model agreement")
    lines.append("")
    lines.append(
        f"- Pairs where ALL models across ALL runs agreed: "
        f"**{stats['cross_model_agreement_pct']}%**"
    )
    lines.append("- Rules-based baseline (by construction): **100%**")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(report.get("interpretation", "<fill in after seeing the numbers>"))
    lines.append("")
    return "\n".join(lines)
