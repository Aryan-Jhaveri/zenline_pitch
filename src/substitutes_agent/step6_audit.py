"""Step 6 - audit trail and vendor-swap diff on Step 5 artifacts.

Post-hoc analysis only. Reads the committed Step 5 outputs and renders:
  - output/step6_audit_trail.md: one section per borderline pair with both
    SKUs' ontology attributes, the score component table, per-model run
    verdicts plus majority, and self/cross-model flags.
  - output/step6_vendor_diff.md: majority-vote-per-model vendor comparison.
  - output/step6_vendor_diff.json: machine-readable diff stats.

No new LLM calls. No new dependencies. Deterministic.

Pair alignment: the set of borderline pairs is whatever Step 5 actually
ran. We read the pair_ids from step5_transcripts.jsonl and look up each
pair's score components by re-running iter_candidate_pairs and filtering
to those pair_ids, so the audit trail and the Step 5 experiment use
identical pairs.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from substitutes_agent.models import OntologyRecord, RunLogEntry, ScoreComponents
from substitutes_agent.step3_classify import iter_candidate_pairs

# Score weights, mirrored from step3_classify (kept private there). Used only
# for the audit trail's component table display.
WEIGHTS: dict[str, float] = {
    "article_type_match": 0.35,
    "colour_similarity": 0.20,
    "usage_match": 0.15,
    "pattern_match": 0.10,
    "material_similarity": 0.15,
    "season_overlap": 0.05,
}

_NONE = "(none)"


def load_transcripts(path: Path) -> dict[str, dict[str, list[str]]]:
    """Return verdicts[pair_id][model_name] = [verdict, verdict, verdict]."""
    verdicts: dict[str, dict[str, list[str]]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = d["pair"]
            model = d["model"]
            verdict = d["verdict"]
            verdicts.setdefault(pid, {}).setdefault(model, []).append(verdict)
    return verdicts


def majority_vote(verdicts: list[str]) -> str:
    """Modal verdict. Ties return 'no' (conservative). Empty list raises."""
    if not verdicts:
        raise ValueError("majority_vote: empty verdict list")
    yes = sum(1 for v in verdicts if v == "yes")
    no = sum(1 for v in verdicts if v == "no")
    return "yes" if yes > no else "no"


def compute_per_model_majority(
    verdicts: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, str]]:
    """Return majorities[pair_id][model_name] = modal verdict per model."""
    return {
        pid: {model: majority_vote(runs) for model, runs in runs_by_model.items()}
        for pid, runs_by_model in verdicts.items()
    }


def compute_vendor_diff(
    majorities: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Summarize how often vendors disagree under majority voting."""
    all_vendors = sorted({v for m in majorities.values() for v in m})
    per_vendor_yes_count: dict[str, int] = {v: 0 for v in all_vendors}
    vendor_split_pairs: list[str] = []
    all_agree = 0
    for pid, model_majorities in majorities.items():
        for v in all_vendors:
            if model_majorities.get(v) == "yes":
                per_vendor_yes_count[v] += 1
        distinct = set(model_majorities.values())
        if len(distinct) <= 1:
            all_agree += 1
        else:
            vendor_split_pairs.append(pid)
    return {
        "total_pairs": len(majorities),
        "all_vendors_agree_count": all_agree,
        "vendor_split_count": len(vendor_split_pairs),
        "vendor_split_pairs": sorted(vendor_split_pairs),
        "per_vendor_yes_count": per_vendor_yes_count,
    }


def _attr_kv(rec: OntologyRecord) -> list[tuple[str, str]]:
    return [
        ("brand", rec.brand_normalized),
        ("article_type", rec.article_type),
        ("master_category", rec.master_category),
        ("base_colour", rec.base_colour),
        ("usage", rec.usage),
        ("gender", rec.gender),
        ("season", rec.season or _NONE),
        ("pattern", rec.pattern or _NONE),
        ("material", rec.material or _NONE),
    ]


def render_audit_trail_md(
    borderline_pairs: list[tuple[str, str, ScoreComponents, float]],
    ontology_by_code: dict[str, OntologyRecord],
    verdicts: dict[str, dict[str, list[str]]],
) -> str:
    """Render the per-pair audit trail as Markdown. No em-dashes."""
    lines: list[str] = []
    lines.append("# Step 6 audit trail")
    lines.append("")
    lines.append(
        "Per-pair audit for the borderline pairs Step 5 ran. Scores and "
        "verdicts come from committed Step 5 artifacts; no new LLM calls."
    )
    lines.append("")

    for sku_a, sku_b, comp, total in borderline_pairs:
        pid = f"{sku_a}|{sku_b}"
        rec_a = ontology_by_code[sku_a]
        rec_b = ontology_by_code[sku_b]
        lines.append(f"## Pair {pid}: {rec_a.product_name} vs {rec_b.product_name}")
        lines.append("")

        lines.append(f"### SKU {sku_a}")
        for k, v in _attr_kv(rec_a):
            lines.append(f"- {k}: {v}")
        lines.append("")

        lines.append(f"### SKU {sku_b}")
        for k, v in _attr_kv(rec_b):
            lines.append(f"- {k}: {v}")
        lines.append("")

        lines.append("### Score components")
        lines.append("| Component | Value | Weight | Weighted |")
        lines.append("| --- | ---: | ---: | ---: |")
        for k, w in WEIGHTS.items():
            val = float(getattr(comp, k))
            lines.append(f"| {k} | {val:.4f} | {w:.2f} | {val * w:.4f} |")
        lines.append(f"| **total** |  |  | **{float(total):.4f}** |")
        lines.append("")

        lines.append("### Verdicts")
        model_runs = verdicts.get(pid, {})
        majorities: dict[str, str] = {}
        for model in sorted(model_runs):
            runs = model_runs[model]
            maj = majority_vote(runs)
            majorities[model] = maj
            lines.append(f"- {model}: {', '.join(runs)} -> majority: {maj}")
        lines.append("")

        flags: list[str] = []
        if any(len(set(runs)) > 1 for runs in model_runs.values()):
            flags.append("self-disagreement")
        if len(set(majorities.values())) > 1:
            flags.append("cross-model split")
        if flags:
            joined = ", ".join(f"[{f}]" for f in flags)
            lines.append(f"**Flags:** {joined}")
        else:
            lines.append("**Flags:** none")
        lines.append("")

    return "\n".join(lines)


def render_vendor_diff_md(
    diff_stats: dict[str, Any],
    majorities: dict[str, dict[str, str]],
) -> str:
    """Render the vendor-swap diff as Markdown. No em-dashes."""
    lines: list[str] = []
    lines.append("# Vendor-swap diff (majority-vote per model)")
    lines.append("")
    lines.append("## Headline stats")
    lines.append(f"- Total pairs: {diff_stats['total_pairs']}")
    lines.append(f"- All vendors agree: {diff_stats['all_vendors_agree_count']}")
    lines.append(f"- Vendor splits: {diff_stats['vendor_split_count']}")
    split_pairs = diff_stats["vendor_split_pairs"]
    if split_pairs:
        lines.append(f"- Vendor-split pairs: {', '.join(split_pairs)}")
    lines.append("")

    lines.append("## Per-vendor yes-count (majority voting)")
    lines.append("| Vendor | Yes-count |")
    lines.append("| --- | ---: |")
    for v, c in diff_stats["per_vendor_yes_count"].items():
        lines.append(f"| {v} | {c} |")
    lines.append("")

    lines.append("## Vendor-split pairs")
    all_vendors = list(diff_stats["per_vendor_yes_count"].keys())
    if not split_pairs:
        lines.append("(none)")
        lines.append("")
    else:
        header = "| Pair | " + " | ".join(all_vendors) + " |"
        sep = "| --- | " + " | ".join("---" for _ in all_vendors) + " |"
        lines.append(header)
        lines.append(sep)
        for pid in split_pairs:
            cells = " | ".join(majorities[pid].get(v, "?") for v in all_vendors)
            lines.append(f"| {pid} | {cells} |")
        lines.append("")

    total = diff_stats["total_pairs"]
    split_count = diff_stats["vendor_split_count"]
    pct = round(100.0 * split_count / total, 1) if total else 0.0
    lines.append("## Reading")
    lines.append("")
    lines.append(
        f"The majority-vote vendor-swap flip rate on this sample is "
        f"{split_count} of {total} pairs ({pct}%). The raw run-level "
        f"cross-model disagreement rate is higher because it counts "
        f"within-model noise as disagreement; taking the modal verdict per "
        f"model collapses most of that noise. The two numbers differ "
        f"because run-level disagreement includes pairs where one model "
        f"flapped on its own three runs while the other two vendors stayed "
        f"consistent. See output/step5_consistency.md for the run-level "
        f"figure."
    )
    lines.append("")
    return "\n".join(lines)


def run_step6(
    ontology_path: Path,
    transcripts_path: Path,
    consistency_json_path: Path,
    audit_md_out: Path,
    vendor_diff_md_out: Path,
    vendor_diff_json_out: Path,
) -> RunLogEntry:
    """Orchestrate Step 6: load inputs, compute, write artifacts. No LLM calls."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    verdicts = load_transcripts(transcripts_path)
    majorities = compute_per_model_majority(verdicts)
    diff_stats = compute_vendor_diff(majorities)

    records = [
        OntologyRecord(**d)
        for d in json.loads(Path(ontology_path).read_text(encoding="utf-8"))
    ]
    ontology_by_code = {r.code: r for r in records}

    consistency = json.loads(Path(consistency_json_path).read_text(encoding="utf-8"))
    if "verdicts" in consistency:
        expected = set(consistency["verdicts"].keys())
        got = set(verdicts.keys())
        if expected != got:
            raise ValueError(
                "Pair id mismatch between transcripts and consistency json: "
                f"only in transcripts={got - expected}, "
                f"only in consistency={expected - got}"
            )

    transcript_ids = set(verdicts.keys())
    borderline_pairs: list[tuple[str, str, ScoreComponents, float]] = []
    for a, b, comp, _ in iter_candidate_pairs(records):
        pid = f"{a.code}|{b.code}"
        if pid in transcript_ids:
            borderline_pairs.append((a.code, b.code, comp, comp.total_score))
    borderline_pairs.sort(key=lambda t: (t[0], t[1]))

    audit_md = render_audit_trail_md(borderline_pairs, ontology_by_code, verdicts)
    vendor_diff_md = render_vendor_diff_md(diff_stats, majorities)

    for out_path, content in (
        (audit_md_out, audit_md),
        (vendor_diff_md_out, vendor_diff_md),
    ):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    jout = Path(vendor_diff_json_out)
    jout.parent.mkdir(parents=True, exist_ok=True)
    jout.write_text(
        json.dumps(diff_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    duration = time.monotonic() - started
    entry = RunLogEntry(
        step="step6_audit",
        started_at=started_at,
        duration_s=round(duration, 4),
        input_rows=len(records),
        output_rows=len(borderline_pairs),
        filters={
            "total_pairs": diff_stats["total_pairs"],
            "vendor_split_count": diff_stats["vendor_split_count"],
            "vendor_split_pairs": diff_stats["vendor_split_pairs"],
        },
        notes="Audit trail and vendor-swap diff on step 5 artifacts. No LLM calls.",
    )
    return entry
